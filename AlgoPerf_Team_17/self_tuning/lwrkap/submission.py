"""Submission by David Tweedle to ML Commons algorithmic-efficiency contest

To briefly summarize this submission: before all reducing the gradients, we approximate the gradients by a low rank approximation, then all reduce.
This allows us to increase the batch size by a factor of N_GPUS.
"""

from typing import Callable, Dict, Iterator, List, Tuple
import collections

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR

import torch.distributed as dist

from absl import logging
import optax
import torch
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import LambdaLR

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()
lrkaState = None
# global variable to store the state of the communication hook
# only used when calling this submission using
# tuning_ruleset=external and using multiple tuning trials
# if only using self tuning this is not needed

import tensorly as tl
tl.set_backend("pytorch")
# used to find low rank approximations of gradient
# using either the cp decomposition
# or randomized_svd decomposition

class LowRankApproximationState:
  """ A class to store all the state information for
      the communication hook
  """

  def __init__(
          self,
          cp,
          svd_rank,
          tucker_rank,
          tol,
          random_state,
          gpu_id,
          n_gpus,
          global_step
          ):
    self.cp = cp
    self.svd_rank = svd_rank
    self.tucker_rank=tucker_rank
    self.tol = tol
    self.random_state = random_state
    self.gpu_id = gpu_id
    self.n_gpus = n_gpus
    self.global_step = global_step

  def __setstate__(self, state):
    self.cp = state['cp']
    self.svd_rank = state['svd_rank']
    self.tucker_rank = state['tucker_rank']
    self.tol = state['tol']
    self.random_state = state['random_state']
    self.gpu_id = state['gpu_id']
    self.n_gpus = state['n_gpus']
    self.global_step = state['global_step']

  def __getstate__(self):
    res = {'cp': self.cp,
           'svd_rank': self.svd_rank,
           'tucker_rank': self.tucker_rank,
           'tol': self.tol,
           'random_state': self.random_state,
           'gpu_id': self.gpu_id,
           'n_gpus': self.n_gpus,
           'global_step': self.global_step
           }
    return res

  def getstep(self):
    return self.global_step

  def setstep(self, step):
    self.global_step = step


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a Nesterov optimizer and a learning rate schedule.
  Returns:
  optimizer state
  optimizer_update_fn
  """
  global lrkaState
  if hyperparameters is None:
    hparams_dict = {'learning_rate': 1.0,
                    'momentum': 0.2,
                    'l2': 5e-4,
                    'cp_rank': 1,
                    'svd_rank': 10,
                    'tucker_rank': 1,
                    'tol': 0.1,
                    'dropout_rate': 0.0,
                    'aux_dropout_rate': 0.0
                    }
    hyperparameters = collections.namedtuple('Hyperparameters', hparams_dict)(**hparams_dict)
  cp = tl.decomposition.CP(rank=hyperparameters.cp_rank,
                           tol=hyperparameters.tol,
                           init='random',
                           n_iter_max=5
                           )
  random_state = int(rng[0])
  if random_state < 0:
      random_state += 2 ** 32
  state = {'cp': cp,
           'tucker_rank': hyperparameters.tucker_rank,
           'svd_rank': hyperparameters.svd_rank,
           'tol': hyperparameters.tol,
           'random_state': random_state,
           'gpu_id': RANK,
           'n_gpus': N_GPUS,
           'global_step': 0
           }
  if lrkaState is None:
    lrkaState = LowRankApproximationState(**state)
    model_params.register_comm_hook(lrkaState, cp_hook)
    # register the communication hook which will
    # approximate the gradient on each gpu
    # then all reduce the results
  else:
    lrkaState.__setstate__(state)
    # if this has been run using num_tuning_trials > 1
    # then we will need to re use the previous communication hook

  base_lr = hyperparameters.learning_rate
  optimizer_state = {
      'optimizer':
          torch.optim.SGD(
              model_params.parameters(),
              lr=base_lr,
              momentum=hyperparameters.momentum,
              weight_decay=hyperparameters.l2),
  }

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
  """
    Returns:
     (new_optimizer_state, update_fn)
     new_params
     new_model_state
    """
  global lrkaState
  lrkaState.setstep(global_step)
  # we need to note the global_step so
  # that on the first step we do not calculate the
  # svd of the gradients
  del current_params_types
  del loss_type
  del eval_results

  assert USE_PYTORCH_DDP
  # this implementation assumes that one is using DDP

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
  loss = summed_loss / n_valid_examples
  loss.backward()

  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(
        current_model.parameters(), max_norm=grad_clip)
  optimizer_state['optimizer'].step()

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
  """
    Gets batch size for workload.
    Args: 
      workload_name (str): Valid workload_name values are: "wmt", "ogbg", 
        "criteo1tb", "fastmri", "imagenet_resnet", "imagenet_vit", 
        "librispeech_deepspeech", "librispeech_conformer".
    Returns:
      int: batch_size 
    Raises:
      ValueError: If workload_name is not handled.
    """
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return N_GPUS * 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return N_GPUS * 256
  elif workload_name == 'librispeech_deepspeech':
    return N_GPUS * 256
  elif workload_name == 'ogbg':
    return N_GPUS * 512
  elif workload_name == 'wmt':
    return N_GPUS * 128
  elif workload_name == 'mnist':
    return N_GPUS * 16
  elif workload_name =='cifar':
    return N_GPUS * 128
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
    Tip:
    If you would just like the next batch from the input queue return next(input_queue).

    Returns:
     batch: next batch of input data
    """
  return next(input_queue)

def cp_hook(state: LowRankApproximationState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
  if state.getstep() > 1:
    for grad in bucket.gradients():
      if len(grad.size()) > 2:
        try:
#          ranks = [min(state.tucker_rank, rank) for rank in grad.size()]
#          decomp = tl.decomposition.partial_tucker(tensor=grad,
#                                     rank=ranks,
#                                     modes=None,
#                                     n_iter_max=5,
#                                     init='svd',
#                                     svd='randomized_svd',
#                                     tol=state.tol,
#                                     random_state=state.random_state
#                                     )
#          grad = tl.tucker_to_tensor(*decomp)
          cp = state.cp
          decomp = cp.fit_transform(tensor=grad)
          grad = tl.cp_to_tensor(decomp)
        except torch._C._LinAlgError as err:
          pass
      elif len(grad.size()) == 2:
        try:
          rank = state.svd_rank if state.svd_rank < grad.size()[0] else grad.size()[0]
          rank = rank if rank < grad.size()[1] else grad.size()[1]
          U,S,Vh = tl.tenalg.svd_interface(
                  matrix=grad,
                  method="randomized_svd",
                  n_eigenvecs=rank,
                  random_state=state.random_state
                  )
          U = U[:,:rank]
          S = S[:rank]
          Vh = Vh[:rank]
          grad = (U * S) @ Vh
        except torch._C._LinAlgError as err:
          pass
      grad.div_(state.n_gpus)
  return dist.all_reduce(bucket.buffer(), async_op=True).get_future().then(lambda fut: fut.value()[0])
