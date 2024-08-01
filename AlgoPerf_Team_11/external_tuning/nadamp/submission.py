"""Template submission module.

See https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#allowed-submissions
and https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#disallowed-submissions
for guidelines.
"""
from typing import Dict, Iterator, List, Tuple, Any, Callable, NamedTuple, Optional, Union
from algorithmic_efficiency import spec
import chex
from flax import jax_utils
import jax
import jax.numpy as jnp
from jax import lax
import optax
import functools
import flax
import flax.linen as nn
import copy
import jraph
import gc

import numpy as np


# LR Schedule

def jax_cosine_warmup(step_hint: int, hyperparameters):
  # Create learning rate schedule.
  warmup_steps = int(hyperparameters.warmup_steps_fraction * step_hint)
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=hyperparameters.learning_rate,
      transition_steps=warmup_steps)
  cosine_steps = max(step_hint - warmup_steps, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=hyperparameters.learning_rate, decay_steps=cosine_steps)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[warmup_steps])
  return schedule_fn

def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)

class ScaleByNadam(NamedTuple):
  """State for the Nadam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: optax.Updates
  nu: optax.Updates

def scale_by_nadam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    power: float = 2.0,
    nesterov: bool = True,
) -> optax.GradientTransformation:
  """Rescale updates according to the NAdam algorithm.

  References:
  There seem to be multiple versions of NAdam. The original version is here
  https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ (the pytorch imp. also
  follows this)

  Current code implements a simpler version with no momentum decay and slightly
  different bias correction terms. The exact description can be found here
  https://arxiv.org/pdf/1910.05446.pdf (Table 1)

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    b2: decay rate for the exponentially weighted average of squared grads.
    eps: term added to the denominator to improve numerical stability.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use bias correction.
    power: the power to use in the preconditioner (the value determines the
      power to which the absolute value of the grads are raised).

  Returns:
    An (init_fn, update_fn) tuple.
  """

  raise_power = (
      jnp.sqrt if power == 2.0 else lambda x: jnp.power(x, 1.0 / power)
  )


  def init_fn(params):    
    def zeros_like_params(params):
      return jax.tree_map(jnp.zeros_like, params)

    jitted_zeros_like = jax.jit(zeros_like_params)

    mu = jitted_zeros_like(params)
    nu = jitted_zeros_like(params)

    return ScaleByNadam(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        nu=nu,
    )

  def update_fn(updates, state, params):
    # Nadam update rule.
    mu = jax.tree_map(lambda g, t: (1-b1)*g + b1*t, updates, state.mu)
    mu_hat = jax.tree_map(lambda g, t: (1-b1)*g + b1*t, updates, mu)

    nu = jax.tree_map(lambda g, t: (1-b2)*(g**2) + b2*t, updates, state.nu)

    count = state.count + jnp.array(1, dtype=jnp.int32)

    if nesterov:
        mu_hat = _bias_correction(mu_hat, b1, count)
    else:
        mu_hat = mu

    nu_hat = _bias_correction(nu, b2, count)

    # New updates.    
    updates = jax.tree_map(
        lambda m, v: m / (raise_power(v + eps_root) + eps), mu_hat, nu_hat
    )

    return updates, ScaleByNadam(
        count=count,
        mu=mu,
        nu=nu,
    )

  return optax.GradientTransformation(init_fn, update_fn)

def instantiate_optimizer(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    debias: bool = True,
    weight_decay: float = 0.0,
    nesterov=True,
    power=2.0,
) -> optax.GradientTransformation:
  """Rescale updates according to the NAdamP algorithm.

  Returns:
    An (init_fn, update_fn) tuple.
  """
  return optax.chain(
    scale_by_nadam(b1=b1, b2=b2, nesterov=nesterov, power=power),
    optax.add_decayed_weights(weight_decay),
    scale_by_learning_rate(learning_rate))


def scale_by_learning_rate(learning_rate, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return optax.scale_by_schedule(lambda count: m * learning_rate(count))
  return optax.scale(m * learning_rate)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates optimizer state and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  target_setting_step_hint = int(hyperparameters.cosine_decay_fraction * workload.step_hint)
  lr_schedule_fn = jax_cosine_warmup(target_setting_step_hint, hyperparameters)

  # Create optimizer.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)

  opt_init_fn, opt_update_fn = instantiate_optimizer(
      learning_rate=lr_schedule_fn,
      b1=hyperparameters.beta1,
      b2=hyperparameters.beta2,
      weight_decay=hyperparameters.weight_decay,
      nesterov=hyperparameters.nesterov,
      power=hyperparameters.power)

  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn

@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, 0, None),
    static_broadcasted_argnums=(0, 1),
    donate_argnums=(2, 3, 4))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       batch,
                       rng,
                       label_smoothing):

  def _loss_fn(params):
    """Loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        update_batch_norm=True)
    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits,
        mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    return summed_loss, (n_valid_examples, new_model_state)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (summed_loss, (n_valid_examples, model_state)), grad = grad_fn(
      current_param_container)
  # Get correct global mean loss and grad.
  (summed_loss, n_valid_examples, grad) = lax.psum(
      (summed_loss, n_valid_examples, grad), axis_name='batch')
  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  updates, optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  current_param_container = optax.apply_updates(current_param_container, updates)
  return optimizer_state, current_param_container, model_state, loss, grad_norm

def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results

  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  if hasattr(hyperparameters, 'label_smoothing'):
    label_smoothing = hyperparameters.label_smoothing
  else:
    label_smoothing = 0.0

  (optimizer_state, current_param_container, model_state, loss, grad_norm) = pmapped_train_step(workload,
                               opt_update_fn,
                               model_state,
                               optimizer_state,
                               current_param_container,
                               batch,
                               per_device_rngs,
                               label_smoothing)

  # Log loss, grad_norm.
  if global_step % 100 == 0 and workload.metrics_logger is not None:
    workload.metrics_logger.append_scalar_metrics(
        {
            'loss': loss[0],
            'grad_norm': grad_norm[0],
        }, global_step)

  return (optimizer_state, opt_update_fn), current_param_container, model_state


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 16
  elif workload_name == 'imagenet_resnet_silu':
    return 512
  elif workload_name == 'imagenet_resnet_gelu':
    return 512
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch