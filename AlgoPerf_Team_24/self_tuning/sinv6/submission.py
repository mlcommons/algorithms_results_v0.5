"""Submission file for task-invariant learned optimizer meta-trained on small tasks (sinv)."""

import functools
from typing import Dict, Iterator, List, Tuple

import os
from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
from .inv6 import Inv

from algorithmic_efficiency import spec
from .utils import cached_jit, load_state


def init_optimizer_state(
    workload: spec.Workload,
    model_params: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    rng: spec.RandomState,
) -> spec.OptimizerState:
    del model_params
    del model_state

    # load pre-trained optimizer
    lopt = Inv(
        lstm_hidden_size=64,
        initial_epsilon=1e-8,
        initial_momentum_decays=(0.9,),
        initial_rms_decays=(0.999,),
        initial_tf_decays=(0.999,),
    )
    theta = lopt.init(rng)
    ckpt_path = os.path.join(os.path.dirname(__file__), "theta.state")
    pretrained_params = load_state(ckpt_path, theta)
    opt = lopt.opt_fn(pretrained_params)

    # create optimizer state
    # num_steps = 0.75 * workload.step_hint  # can be tuned?
    num_steps = workload.step_hint
    params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes)
    opt_state = cached_jit(opt.init, static_argnames=("num_steps",))(
        params_zeros_like, model_state=None, num_steps=num_steps
    )
    opt_update_fn = opt.update
    return jax_utils.replicate(opt_state), opt_update_fn


@functools.partial(
    jax.pmap,
    axis_name="batch",
    in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 1),
    donate_argnums=(2, 3, 4),
)
def pmapped_train_step(
    workload,
    opt_update_fn,
    model_state,
    optimizer_state,
    current_param_container,
    batch,
    rng,
    grad_clip,
    label_smoothing,
):
    # ---------------- reference baseline code  ---------------- #
    def _loss_fn(params):
        """Loss function used for training."""
        logits, new_model_state = workload.model_fn(
            params, batch, model_state, spec.ForwardPassMode.TRAIN, rng, update_batch_norm=True
        )
        loss_dict = workload.loss_fn(
            label_batch=batch["targets"],
            logits_batch=logits,
            mask_batch=batch.get("weights"),
            label_smoothing=label_smoothing,
        )
        summed_loss = loss_dict["summed"]
        n_valid_examples = loss_dict["n_valid_examples"]
        return summed_loss, (n_valid_examples, new_model_state)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(current_param_container)
    # Get correct global mean loss and grad.
    (summed_loss, n_valid_examples, grad) = lax.psum(
        (summed_loss, n_valid_examples, grad), axis_name="batch"
    )
    loss = summed_loss / n_valid_examples
    grad = jax.tree_map(lambda x: x / n_valid_examples, grad)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))
    # -------------------------------------------------------- #

    # lopt update
    updated_params, new_optimizer_state = opt_update_fn(
        optimizer_state, grad, current_param_container, loss
    )
    return new_optimizer_state, updated_params, new_model_state, loss, grad_norm


def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    batch: Dict[str, spec.Tensor],
    loss_type: spec.LossType,
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState,
) -> spec.UpdateReturn:
    """Return (updated_optimizer_state, updated_params, updated_model_state)."""
    del current_params_types
    del loss_type
    del eval_results

    optimizer_state, opt_update_fn = optimizer_state
    per_device_rngs = jax.random.split(rng, jax.local_device_count())
    if hasattr(hyperparameters, "label_smoothing"):
        label_smoothing = hyperparameters.label_smoothing
    else:
        label_smoothing = 0.0
    if hasattr(hyperparameters, "grad_clip"):
        grad_clip = hyperparameters.grad_clip
    else:
        grad_clip = None
    outputs = pmapped_train_step(
        workload,
        opt_update_fn,
        model_state,
        optimizer_state,
        current_param_container,
        batch,
        per_device_rngs,
        grad_clip,
        label_smoothing,
    )
    new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs

    # Log loss, grad_norm.
    if global_step % 100 == 0 and workload.metrics_logger is not None:
        workload.metrics_logger.append_scalar_metrics(
            {
                "loss": loss[0],
                "grad_norm": grad_norm[0],
            },
            global_step,
        )
    return (new_optimizer_state, opt_update_fn), new_params, new_model_state


def get_batch_size(workload_name):
    # Return the global batch size.
    if workload_name == "criteo1tb":
        return 262_144
    elif workload_name == "fastmri":
        return 32
    elif workload_name == "imagenet_resnet":
        return 1024
    elif workload_name == "imagenet_vit":
        return 1024
    elif workload_name == "librispeech_conformer":
        return 256
    elif workload_name == "librispeech_deepspeech":
        return 256
    elif workload_name == "ogbg":
        return 512
    elif workload_name == "wmt":
        return 128
    elif workload_name == "mnist":
        return 16
    else:
        raise ValueError(f"Unsupported workload name: {workload_name}.")


def data_selection(
    workload: spec.Workload,
    input_queue: Iterator[Dict[str, spec.Tensor]],
    optimizer_state: spec.OptimizerState,
    current_param_container: spec.ParameterContainer,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparameters,
    global_step: int,
    rng: spec.RandomState,
) -> Dict[str, spec.Tensor]:
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
