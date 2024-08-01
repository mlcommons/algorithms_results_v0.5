"""Submission file for an NAdamW optimizer with warmup+cosine LR in PyTorch."""

import math
from typing import Dict, Iterator, List, Tuple

from collections import deque
from itertools import islice

import torch
from torch import Tensor
import torch.distributed.nn as dist_nn

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP = pytorch_setup()[0]

# Modified from github.com/pytorch/pytorch/blob/v1.12.1/torch/optim/adamw.py.
class NAdamW(torch.optim.Optimizer):
  r"""Implements NAdamW algorithm.

    See Table 1 in https://arxiv.org/abs/1910.05446 for the implementation of
    the NAdam algorithm (there is also a comment in the code which highlights
    the only difference of NAdamW and AdamW).
    For further details regarding the algorithm we refer to
    `Decoupled Weight Decay Regularization`_.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
          parameter groups
      lr (float, optional): learning rate (default: 1e-3)
      betas (Tuple[float, float], optional): coefficients used for computing
          running averages of gradient and its square (default: (0.9, 0.999))
      eps (float, optional): term added to the denominator to improve
          numerical stability (default: 1e-8)
      weight_decay (float, optional): weight decay coefficient (default: 1e-2)
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
  """

  def __init__(self,
               params,
               lr=1e-3,
               betas=(0.9, 0.999),
               eps=1e-8,
               weight_decay=1e-2):
    if not 0.0 <= lr:
      raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= eps:
      raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
    if not 0.0 <= weight_decay:
      raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    defaults = {
        'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay
    }
    super().__init__(params, defaults)

  def __setstate__(self, state):
    super().__setstate__(state)
    state_values = list(self.state.values())
    step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
        state_values[0]['step'])
    if not step_is_tensor:
      for s in state_values:
        s['step'] = torch.tensor(float(s['step']))

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
              and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is None:
          continue
        params_with_grad.append(p)
        if p.grad.is_sparse:
          raise RuntimeError('NAdamW does not support sparse gradients')
        grads.append(p.grad)

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = torch.tensor(0.)
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)

        exp_avgs.append(state['exp_avg'])
        exp_avg_sqs.append(state['exp_avg_sq'])
        state_steps.append(state['step'])

      nadamw(
          params_with_grad,
          grads,
          exp_avgs,
          exp_avg_sqs,
          state_steps,
          beta1=beta1,
          beta2=beta2,
          lr=group['lr'],
          weight_decay=group['weight_decay'],
          eps=group['eps'])

    return loss


def nadamw(params: List[Tensor],
           grads: List[Tensor],
           exp_avgs: List[Tensor],
           exp_avg_sqs: List[Tensor],
           state_steps: List[Tensor],
           beta1: float,
           beta2: float,
           lr: float,
           weight_decay: float,
           eps: float) -> None:
  r"""Functional API that performs NAdamW algorithm computation.
    See NAdamW class for details.
  """

  if not all(isinstance(t, torch.Tensor) for t in state_steps):
    raise RuntimeError(
        'API has changed, `state_steps` argument must contain a list of' +
        ' singleton tensors')

  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step_t = state_steps[i]

    # Update step.
    step_t += 1

    # Perform stepweight decay.
    param.mul_(1 - lr * weight_decay)

    # Decay the first and second moment running average coefficient.
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # Only difference between NAdamW and AdamW in this implementation.
    # The official PyTorch implementation of NAdam uses a different algorithm.
    # We undo these ops later on, which could cause numerical issues but saves
    # us from having to make an extra copy of the gradients.
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

    step = step_t.item()

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    step_size = lr / bias_correction1

    bias_correction2_sqrt = math.sqrt(bias_correction2)
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

    param.addcdiv_(exp_avg, denom, value=-step_size)
    exp_avg.sub_(grad, alpha=1 - beta1).div_(beta1)


class WarmCosine(object):
  def __init__(self, optimizer, lr_min, lr_max, warmup_steps, T):
    self.optimizer = optimizer
    self.lr_min = lr_min
    self.lr_max = lr_max
    self.warmup_steps = warmup_steps
    self.T = T
    self.t = 0

  def schedule(self, t):
    if t <= self.warmup_steps:
      return self.lr_min + (self.lr_max-self.lr_min)/self.warmup_steps * t
    elif t <= self.T:
      return self.lr_min + 0.5 * (self.lr_max-self.lr_min) * (1 + math.cos((t-self.warmup_steps) * math.pi / (self.T-self.warmup_steps)))
    return self.lr_min

  def step(self):
    self.t += 1
    # set LR in optimizer
    lr = self.schedule(self.t)
    for group in self.optimizer.param_groups:
      group["lr"] = lr

  def state_dict(self):
    return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)


class LAWA():
  def __init__(self, hyperparameters, workload) -> None:
    self.prev_params = None
    self.maxlen = int(hyperparameters.k)
    self.queue = deque(maxlen=self.maxlen)
    self.local_step = torch.tensor(0.)

    self.lawa_start_step = math.ceil(workload.step_hint * hyperparameters.lawa_start_factor)
    self.lawa_interval = math.ceil(workload.step_hint * hyperparameters.lawa_interval_scaling)

    time_per_step = workload.max_allowed_runtime_sec / workload.step_hint
    steps_per_eval = workload.eval_period_time_sec / time_per_step

    # number of steps in inner loop
    self.steps_per_call = math.ceil(steps_per_eval * hyperparameters.lawa_inner_steps_frac)

  def update_prev(self, params):
    self.prev_params = [p.detach().cpu() for p in params]

  def queue_append(self, params):
    self.queue.append([p.detach().cpu() for p in params])

  def queue_full(self):
    return (len(self.queue)==self.maxlen)

  def queue_avg(self):
    k = float(self.maxlen)

    # Initialize avg with first element of the queue
    q_avg = [p.clone().div_(k) for p in self.queue[0]] # self.queue[0] is already on cpu!

    # Loop over queue and update avg
    for chkpts in islice(self.queue, 1, None):
      for p_avg,p in zip(q_avg, chkpts):
        p_avg.add_(p/k)

    return q_avg
  
  def state_dict(self):
    return {key: value for key, value in self.__dict__.items()}

  def load_state_dict(self, state_dict):
    self.__dict__.update(state_dict)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a NAdamW optimizer and a learning rate schedule."""
  del model_state
  del rng

  optimizer_state = {
      'optimizer':
          NAdamW(
              model_params.parameters(),
              lr=hyperparameters.learning_rate,
              betas=(1.0 - hyperparameters.one_minus_beta1,
                     hyperparameters.beta2),
              eps=1e-8,
              weight_decay=hyperparameters.weight_decay),
      'lawa': LAWA(hyperparameters, workload),
  }

  optimizer_state['scheduler'] = WarmCosine(
      optimizer_state['optimizer'], 
      lr_min = 1e-10, 
      lr_max = hyperparameters.learning_rate, 
      warmup_steps = int(hyperparameters.warmup_factor * workload.step_hint), 
      T = workload.step_hint)

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
  del global_step

  current_model = current_param_container
  lawa = optimizer_state['lawa']

  # Lawa hyperparams
  lawa_start_step = lawa.lawa_start_step
  lawa_interval = lawa.lawa_interval
  steps_per_call = lawa.steps_per_call
  
  # Actual optimization step
  local_step = lawa.local_step

  # Discard average and load previous params
  if local_step > lawa_start_step and lawa.queue_full():
    for p,p_old in zip(current_model.parameters(), lawa.prev_params):
      p.data.copy_(p_old.data)

  # Internal loop
  for _ in range(steps_per_call):

    current_model.train()
    optimizer_state['optimizer'].zero_grad()

    batch_i = next(batch)

    logits_batch, new_model_state = workload.model_fn(
        params=current_model,
        augmented_and_preprocessed_input_batch=batch_i,
        model_state=model_state,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=True)

    label_smoothing = (
        hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                  'label_smoothing') else 0.0)

    loss_dict = workload.loss_fn(
        label_batch=batch_i['targets'],
        logits_batch=logits_batch,
        mask_batch=batch_i.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    if USE_PYTORCH_DDP:
      # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
      summed_loss = dist_nn.all_reduce(summed_loss)
      n_valid_examples = dist_nn.all_reduce(n_valid_examples)
    loss = summed_loss / n_valid_examples

    del(logits_batch)
    loss.backward()

    optimizer_state['optimizer'].step()
    optimizer_state['scheduler'].step()

    # Update queue
    if local_step >= lawa_start_step and \
        (local_step-lawa_start_step) % lawa_interval == 0:
      lawa.queue_append(current_model.parameters())

    # Update local_step
    local_step.add_(1)

  # Save previous parameters
  if local_step >= lawa_start_step:
    lawa.update_prev(current_model.parameters())

    # Load avg into model
    if lawa.queue_full():
      avg = lawa.queue_avg()
      for p, p_avg in zip(current_model.parameters(), avg):
        p.data.copy_(p_avg.data)

  return (optimizer_state, current_model, new_model_state)


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 512
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
  # do not return a batch here,
  # instead draw the batch inside the inner loop directly
  return input_queue
