"""Submission file for an NAdamW optimizer with warmup+cosine LR in Jax."""

import functools

# isort: off
# We have to turn off isort here to resolve a conflict between isort and yapf.
from typing import (Any,
                    Callable,
                    Dict,
                    Iterator,
                    List,
                    NamedTuple,
                    Optional,
                    Tuple,
                    Union)
# isort: on

import chex
from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec
from submissions.submissions_algorithms_v0_5.AlgoPerf_Team_16.external_tuning.caspr_adaptive.caspr_adaptive_helper import efficient_caspr_adaptive_full_matrix_dist_inv_optimized

from flax import struct


_GRAD_CLIP_EPS = 1e-6



def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """CASPR Adaptive optimizer with nested learning rate schedule"""
  del model_params
  del model_state
  del rng

  def jax_cosine_warmup(step_hint: int, hyperparameters):
    # Create learning rate schedule.
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=hyperparameters.learning_rate,
        transition_steps=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=hyperparameters.learning_rate, decay_steps=cosine_steps)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps])
    return schedule_fn

  def jax_cosine_warmup_nested_learning_rate(step_hint: int, step_hint2: int, step_hint3: int, hyperparameters):
    # Create learning rate schedule.
    outer_cosine = jax_cosine_warmup(step_hint3, hyperparameters)
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    warmup_fn = optax.linear_schedule(
        init_value=0.,
        end_value=hyperparameters.learning_rate,
        transition_steps=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=hyperparameters.learning_rate, decay_steps=cosine_steps)
    cosine_steps2 = step_hint2-step_hint
    cosine_steps3 = step_hint3-step_hint2
    cosine_fn2 = optax.cosine_decay_schedule(init_value=outer_cosine(step_hint), decay_steps=cosine_steps2)
    cosine_fn3 = optax.cosine_decay_schedule(init_value=outer_cosine(step_hint2), decay_steps=cosine_steps3)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn, cosine_fn2, cosine_fn3], boundaries=[warmup_steps, step_hint, step_hint2])
    return schedule_fn

  # Create optimizer + LR schedule.
  lr_schedule_fn = jax_cosine_warmup_nested_learning_rate(workload.step_hint*0.53571619898, workload.step_hint*0.75,
                                      workload.step_hint , hyperparameters)
  opt_init_fn, opt_update_fn = efficient_caspr_adaptive_full_matrix_dist_inv_optimized(
       lr_schedule_fn,
       b1=1.0 - hyperparameters.one_minus_beta1,
       b2=hyperparameters.beta2,
       eps=1e-8,
       matrix_epsilon=1e-6,
       eps_root=0.0,
       block_size=1024,
       preconditioning_compute_steps=20,
       start_preconditioning_step=101,
       exponent_override=0,
       nesterov=True,
       caspr_p=-1,
       relative_epsilon=True,
       inverse_type="coupled newton",
       error_tolerance=0.1,
       weight_decay=hyperparameters.weight_decay,
       global_grafting=False,
       batch_axis_name='batch'
     )
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn


@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 1),
    donate_argnums=(2, 3, 4))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       batch,
                       rng,
                       grad_clip,
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
  (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
      current_param_container)
  # Get correct global mean loss and grad.
  (summed_loss, n_valid_examples, grad) = lax.psum(
      (summed_loss, n_valid_examples, grad), axis_name='batch')
  loss = summed_loss / n_valid_examples
  grad = jax.tree_map(lambda x: x / n_valid_examples, grad)

  grad_norm = jnp.sqrt(
      sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

  if grad_clip is not None:
    grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)

  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params, new_model_state, loss, grad_norm




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
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None
  
  outputs = pmapped_train_step(workload,
                               opt_update_fn,
                               model_state,
                               optimizer_state,
                               current_param_container,
                               batch,
                               per_device_rngs,
                               grad_clip,
                               label_smoothing)
  new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs
  
  if global_step % 100 == 0 and workload.metrics_logger is not None:
    
      
    workload.metrics_logger.append_scalar_metrics(
        {
            'loss': loss[0],
            'grad_norm': grad_norm[0]
        }, global_step)
  return (new_optimizer_state, opt_update_fn), new_params, new_model_state


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
