from typing import Iterator, List, Tuple, Dict

from absl import logging
import torch
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch.workload import \
    Criteo1TbDlrmSmallWorkload
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.workload import \
    FastMRIWorkload
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.workload import \
    ImagenetResNetWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_pytorch.workload import \
    ImagenetVitWorkload
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.workload import \
    LibriSpeechConformerWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_pytorch.workload import \
    LibriSpeechDeepSpeechWorkload
from algorithmic_efficiency.workloads.mnist.mnist_pytorch.workload import \
    MnistWorkload
from algorithmic_efficiency.workloads.ogbg.ogbg_pytorch.workload import \
    OgbgWorkload
from algorithmic_efficiency.workloads.wmt.wmt_pytorch.workload import \
    WmtWorkload
from submissions.submissions_algorithms_v0_5.AlgoPerf_Team_6.external_tuning.ggT.optimizer import \
    LocalOptimizer_GGT

USE_PYTORCH_DDP = pytorch_setup()[0]


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a NAdamW optimizer and a learning rate schedule."""
  del model_state
  del rng

  batch_size = get_batch_size(workload)
  lr = hyperparameters.learning_rate * batch_size / 128
  optimizer_state = {
      'optimizer':
          LocalOptimizer_GGT(
              model_params,
              lr=lr,
              momentum=hyperparameters.momentum,
              damping=hyperparameters.damping,
              beta2=hyperparameters.beta2,
              weight_decay=hyperparameters.weight_decay,
              T=hyperparameters.T,
              lr_cov=hyperparameters.lr_cov,
              using_matrix_norm=True,
              warmup_factor=hyperparameters.warmup_factor,
              batch_size=batch_size,
          )
  }

  def pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
    warmup_steps = int(hyperparameters.warmup_factor * step_hint)
    warmup = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
    cosine_steps = max(step_hint - warmup_steps, 1)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps])

  optimizer_state['scheduler'] = pytorch_cosine_warmup(
      workload.step_hint * hyperparameters.step_hint_factor,
      hyperparameters,
      optimizer_state['optimizer'])

  return optimizer_state


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

  current_model = current_param_container
  current_model.train()
  optimizer_state['optimizer'].zero_grad()

  logits_batch, new_model_state = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch=batch,
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

  label_smoothing = (
      hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                 'label_smoothing') else 0.0)
  if hasattr(hyperparameters, 'grad_clip'):
    grad_clip = hyperparameters.grad_clip
  else:
    grad_clip = None

  loss_dict = workload.loss_fn(
      label_batch=batch['targets'],
      logits_batch=logits_batch,
      mask_batch=batch.get('weights'),
      label_smoothing=label_smoothing)
  summed_loss = loss_dict['summed']
  n_valid_examples = loss_dict['n_valid_examples']
  if USE_PYTORCH_DDP:
    # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
    summed_loss = dist_nn.all_reduce(summed_loss)
    n_valid_examples = dist_nn.all_reduce(n_valid_examples)
  loss = summed_loss / n_valid_examples

  loss.backward()

  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(
        current_model.parameters(), max_norm=grad_clip)
  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()

  # Log training metrics - loss, grad_norm, batch_size.
  if global_step <= 100 or global_step % 500 == 0:
    with torch.no_grad():
      parameters = [p for p in current_model.parameters() if p.grad is not None]
      grad_norm = torch.norm(
          torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
    if workload.metrics_logger is not None:
      workload.metrics_logger.append_scalar_metrics(
          {
              'loss': loss.item(),
              'grad_norm': grad_norm.item(),
          }, global_step)
    logging.info('%d) loss = %0.3f, grad_norm = %0.3f',
                 global_step,
                 loss.item(),
                 grad_norm.item())

  return (optimizer_state, current_param_container, new_model_state)


def get_batch_size(workload_name):
  if not isinstance(workload_name, str):
    if isinstance(workload_name, OgbgWorkload):
      return get_batch_size('ogbg')
    elif isinstance(workload_name, WmtWorkload):
      return get_batch_size('wmt')
    elif isinstance(workload_name, ImagenetResNetWorkload):
      return get_batch_size('imagenet_resnet')
    elif isinstance(workload_name, ImagenetVitWorkload):
      return get_batch_size('imagenet_vit')
    elif isinstance(workload_name, LibriSpeechConformerWorkload):
      return get_batch_size('librispeech_conformer')
    elif isinstance(workload_name, LibriSpeechDeepSpeechWorkload):
      return get_batch_size('librispeech_deepspeech')
    elif isinstance(workload_name, Criteo1TbDlrmSmallWorkload):
      return get_batch_size('criteo1tb')
    elif isinstance(workload_name, FastMRIWorkload):
      return get_batch_size('fastmri')
    elif isinstance(workload_name, MnistWorkload):
      return get_batch_size('mnist')
    else:
      raise ValueError
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
