from collections import defaultdict
import math
from typing import Optional, Tuple, List, Dict

from absl import logging
import numpy as np
import torch
from torch import nn
from torch import optim


# pylint: disable=C0103
class LocalOptimizer_GGT(optim.Optimizer):

  def __init__(
      self,
      model: nn.Module,
      lr: float = 0.001,
      momentum: float = 0.9,
      damping: float = 0.001,
      beta2: float = 0.5,
      weight_decay: float = 0.0,
      T: int = 10,
      batch_averaged: bool = True,
      lr_cov: float = 1e-2,
      using_matrix_norm: bool = True,
      warmup_factor: int = 1,
      batch_size: Optional[int] = None,
      cast_dtype: torch.dtype = torch.float32,
  ):
    if lr < 0.0:
      raise ValueError(f"Invalid learning rate: {lr}")
    if lr_cov < 0.0:
      raise ValueError(f"Invalid learning rate for cov: {lr_cov}")
    if beta2 < 0.0:
      raise ValueError(f"Invalid beta2: {lr_cov}")
    if momentum < 0.0:
      raise ValueError(f"Invalid momentum value: {momentum}")
    if weight_decay < 0.0:
      raise ValueError(f"Invalid weight_decay value: {weight_decay}")
    defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

    self.using_matrix_norm = using_matrix_norm
    self.warmup_factor = warmup_factor
    self.is_print = False
    self.lr_cov = lr_cov
    self.damping = damping
    self.beta2 = beta2
    self.cast_dtype = cast_dtype

    self.batch_averaged = batch_averaged
    if batch_averaged:
      assert batch_size is not None
      self.batch_size = batch_size

    self.params_list = defaultdict(list)
    params = self._prepare_model(model)
    super().__init__(params, defaults)

    self.steps = 0
    self.next_step = 0
    self.scaling: Dict[str, float] = {}

    self.precond_B_blocks: Dict[str, torch.Tensor] = {}
    self.precond_m_B_blocks: Dict[str, torch.Tensor] = {}
    self.precond_BBt_blocks: Dict[str, torch.Tensor] = {}

    self.T = T
    self.log_scaling = {}

    self.max_info = {}  #debug
    self.is_max_update = set()
    self.cur_info = {}  #debug

  def record_info(
      self,
      tensor: torch.Tensor,
      name: str,
      ignore_nan: bool = False,
  ) -> Tuple[float, bool]:
    max_t = torch.max(torch.abs(tensor)).item()
    key = "max_{name}"
    item = self.max_info.get(key, None)
    is_update = False
    if item is None or item < max_t:
      self.max_info[key] = max_t
      self.is_max_update.add(key)
      is_update = True

    if ignore_nan:
      pass
    else:
      flag = torch.isfinite(tensor).all()
      if not flag:
        logging.info("warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info(f"{name}, {max_t}, {tensor}")
      assert flag
    return max_t, is_update

  def full_record_info(
      self,
      stats: Tuple[str, torch.Tensor],
      mat: Tuple[str, torch.Tensor],
  ):
    (stats_name, stats_info) = stats  #x or g
    (mat_name, mat_info) = mat  #A or B

    max_stats, is_update = self.record_info(
        stats_info @ mat_info, '%s%s'%(stats_name,mat_name))
    if is_update:
      max_stats = torch.max(torch.abs(stats_info)).item()
      norm_stats = stats_info.norm().item()
      self.cur_info[f"{stats_name}_max"] = max_stats
      self.cur_info[f"{stats_name}_norm"] = norm_stats

      max_mat = torch.max(torch.abs(mat_info)).item()
      norm_mat = mat_info.norm().item()
      self.cur_info[f"{mat_name}_max"] = max_mat
      self.cur_info[f"{mat_name}_norm"] = norm_mat

  def _prepare_model(self, model: nn.Module) -> List[nn.Parameter]:
    named_params = model.named_parameters()
    discount_modules = {
        "BatchNorm2d", "LayerNorm2d", "GlobalResponseNorm", "LayerNorm"
    }
    params = []
    tmp = {}
    self.discount_factor = set()
    for name, param in named_params:
      if param.requires_grad:
        key = name
        info = self.params_list.setdefault(key, [])
        info.append(param)
        params.append(param)
        tmp.setdefault(param.data_ptr(), key)

    for module in model.modules():
      classname = module.__class__.__name__
      if classname in discount_modules:
        for name, param in module.named_parameters():
          if param.requires_grad and param.data_ptr() in tmp:
            self.discount_factor.add(tmp[param.data_ptr()])

    #logging.info(f"Discount factor: {self.discount_factor}")
    return params

  def get_H_values(
      self,
      key: str,
      G: torch.Tensor,
      precond_B_blocks: Dict[str, torch.Tensor],
      damping: float,
      scaling: float = 1.0,
  ) -> Dict[str, torch.Tensor]:
    results: Dict[str, torch.Tensor] = {}

    name1 = f"{key}_dim-1"
    name2 = f"{key}_dim-2"
    name3 = f"{key}_dim-3"
    name4 = f"{key}_dim-4"

    # we assume that G = torch.squeeze(p_grad)
    if G.dim() == 1:
      results[name1] = (damping / 2.0 *
                        precond_B_blocks[name1].t()) @ precond_B_blocks[name1]
      half = G.view(1, -1) @ precond_B_blocks[name1]
      results[name1].add_(half.t() @ half, alpha=1.0 / 2.0)

      k = torch.tensor(range(len(results[name1])))
      results[name1][k,
                     k] = torch.diagonal(results[name1]) - 1.0 / (2.0 * scaling)

    elif G.dim() == 2:
      B1 = precond_B_blocks[name1] / math.sqrt(len(precond_B_blocks[name1]))
      B2 = precond_B_blocks[name2] / math.sqrt(len(precond_B_blocks[name2]))

      results[name1] = B1.t() @ B1
      results[name2] = B2.t() @ B2
      tr_BBt1 = torch.trace(results[name1])
      tr_BBt2 = torch.trace(results[name2])
      results[name1].mul_(
          (damping * len(precond_B_blocks[name1])) * tr_BBt2 / 2.0)
      results[name2].mul_(
          (damping * len(precond_B_blocks[name2])) * tr_BBt1 / 2.0)

      tmp = B2.t() @ G @ B1
      results[name1].add_(tmp.t() @ tmp, alpha=len(results[name1]) / 2.0)
      results[name2].add_(tmp @ tmp.t(), alpha=len(results[name2]) / 2.0)

      k = torch.tensor(range(len(results[name1])))
      results[name1][k,
                     k] = torch.diagonal(results[name1]) - 1.0 / (2.0 * scaling)

      k = torch.tensor(range(len(results[name2])))
      results[name2][k,
                     k] = torch.diagonal(results[name2]) - 1.0 / (2.0 * scaling)

    elif G.dim() == 3:  # 3d tensor
      B1 = precond_B_blocks[name1] / math.sqrt(len(precond_B_blocks[name1]))
      B2 = precond_B_blocks[name2] / math.sqrt(len(precond_B_blocks[name2]))
      B3 = precond_B_blocks[name3] / math.sqrt(len(precond_B_blocks[name3]))
      results[name1] = B1.t() @ B1
      results[name2] = B2.t() @ B2
      results[name3] = B3.t() @ B3
      tr_BBt1 = torch.trace(results[name1])
      tr_BBt2 = torch.trace(results[name2])
      tr_BBt3 = torch.trace(results[name3])
      results[name1].mul_(
          (len(precond_B_blocks[name1]) * damping) * tr_BBt2 * tr_BBt3 / 2.0)
      results[name2].mul_(
          (len(precond_B_blocks[name2]) * damping) * tr_BBt1 * tr_BBt3 / 2.0)
      results[name3].mul_(
          (len(precond_B_blocks[name3]) * damping) * tr_BBt1 * tr_BBt2 / 2.0)

      tmp_common = torch.einsum('pi,ijk->pjk', B3.t(), G)
      tmp1_half = torch.einsum('pjk,jq->pqk', tmp_common, B2)
      tmp1 = torch.einsum('pqm,pqk->mk', tmp1_half, tmp1_half)
      results[name1].add_(
          precond_B_blocks[name1].t() @ tmp1 @ precond_B_blocks[name1],
          alpha=1.0 / 2.0)
      k = torch.tensor(range(len(results[name1])))
      results[name1][k,
                     k] = torch.diagonal(results[name1]) - 1.0 / (2.0 * scaling)

      tmp2_half = torch.einsum('pjk,km->pjm', tmp_common, B1)
      tmp2 = torch.einsum('pqm,pjm->qj', tmp2_half, tmp2_half)
      results[name2].add_(
          precond_B_blocks[name2].t() @ tmp2 @ precond_B_blocks[name2],
          alpha=1.0 / 2.0)
      k = torch.tensor(range(len(results[name2])))
      results[name2][k,
                     k] = torch.diagonal(results[name2]) - 1.0 / (2.0 * scaling)

      tmp_remaining = torch.einsum('ijk,jq->iqk', G, B2)
      tmp3_half = torch.einsum('iqk,km->iqm', tmp_remaining, B1)
      tmp3 = torch.einsum('pqm,iqm->pi', tmp3_half, tmp3_half)
      results[name3].add_(
          precond_B_blocks[name3].t() @ tmp3 @ precond_B_blocks[name3],
          alpha=1.0 / 2.0)
      k = torch.tensor(range(len(results[name3])))
      results[name3][k,
                     k] = torch.diagonal(results[name3]) - 1.0 / (2.0 * scaling)

    elif G.dim() == 4:  # 4d tensor
      B1 = precond_B_blocks[name1] / math.sqrt(len(precond_B_blocks[name1]))
      B2 = precond_B_blocks[name2] / math.sqrt(len(precond_B_blocks[name2]))
      B3 = precond_B_blocks[name3] / math.sqrt(len(precond_B_blocks[name3]))
      B4 = precond_B_blocks[name4] / math.sqrt(len(precond_B_blocks[name4]))
      results[name1] = B1.t() @ B1
      results[name2] = B2.t() @ B2
      results[name3] = B3.t() @ B3
      results[name4] = B4.t() @ B4
      tr_BBt1 = torch.trace(results[name1])
      tr_BBt2 = torch.trace(results[name2])
      tr_BBt3 = torch.trace(results[name3])
      tr_BBt4 = torch.trace(results[name4])
      results[name1].mul_((len(precond_B_blocks[name1]) * damping) * tr_BBt2 *
                          tr_BBt3 * tr_BBt4 / 2.0)
      results[name2].mul_((len(precond_B_blocks[name2]) * damping) * tr_BBt1 *
                          tr_BBt3 * tr_BBt4 / 2.0)
      results[name3].mul_((len(precond_B_blocks[name3]) * damping) * tr_BBt1 *
                          tr_BBt2 * tr_BBt4 / 2.0)
      results[name4].mul_((len(precond_B_blocks[name4]) * damping) * tr_BBt1 *
                          tr_BBt2 * tr_BBt3 / 2.0)

      tmp_common = torch.einsum('pi,ijkl->pjkl', B4.t(), G)
      tmp_a = torch.einsum('pjkl,jq->pqkl', tmp_common, B3)
      tmp1_half = torch.einsum('pqkl,km->pqml', tmp_a, B2)
      tmp1 = torch.einsum('pqmw,pqml->wl', tmp1_half, tmp1_half)
      results[name1].add_(
          precond_B_blocks[name1].t() @ tmp1 @ precond_B_blocks[name1],
          alpha=1.0 / 2.0)
      k = torch.tensor(range(len(results[name1])))
      results[name1][k,
                     k] = torch.diagonal(results[name1]) - 1.0 / (2.0 * scaling)

      tmp2_half = torch.einsum('pqkl,lw->pqkw', tmp_a, B1)
      tmp2 = torch.einsum('pqmw,pqkw->mk', tmp2_half, tmp2_half)
      results[name2].add_(
          precond_B_blocks[name2].t() @ tmp2 @ precond_B_blocks[name2],
          alpha=1.0 / 2.0)
      k = torch.tensor(range(len(results[name2])))
      results[name2][k,
                     k] = torch.diagonal(results[name2]) - 1.0 / (2.0 * scaling)

      tmp_b = torch.einsum('pjkl,km->pjml', tmp_common, B2)
      tmp3_half = torch.einsum('pjml,lw->pjmw', tmp_b, B1)
      tmp3 = torch.einsum('pqmw,pjmw->qj', tmp3_half, tmp3_half)
      results[name3].add_(
          precond_B_blocks[name3].t() @ tmp3 @ precond_B_blocks[name3],
          alpha=1.0 / 2.0)
      k = torch.tensor(range(len(results[name3])))
      results[name3][k,
                     k] = torch.diagonal(results[name3]) - 1.0 / (2.0 * scaling)

      tmp_remaining = torch.einsum('ijkl,jq->iqkl', G, B3)
      tmp_c = torch.einsum('iqkl,km->iqml', tmp_remaining, B2)
      tmp4_half = torch.einsum('iqml,lw->iqmw', tmp_c, B1)
      tmp4 = torch.einsum('pqmw,iqmw->pi', tmp4_half, tmp4_half)
      results[name4].add_(
          precond_B_blocks[name4].t() @ tmp4 @ precond_B_blocks[name4],
          alpha=1.0 / 2.0)
      k = torch.tensor(range(len(results[name4])))
      results[name4][k,
                     k] = torch.diagonal(results[name4]) - 1.0 / (2.0 * scaling)

    else:
      raise NotImplementedError

    return results

  def _update_inv(self, m: nn.Module, G: torch.Tensor):
    """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
    damping = self.damping

    res = self.get_H_values(m, G, self.precond_B_blocks, damping)
    scaling = 1.0
    self.scaling[m] = 1.0

    if self.using_matrix_norm:
      lr1 = self.lr_cov
    else:
      if self.steps < 50 * self.warmup_factor:
        step_lr_cov = self.lr_cov / 10000.0
      elif self.steps < 100 * self.warmup_factor:
        step_lr_cov = self.lr_cov / 1000.0
      elif self.steps < 150 * self.warmup_factor:
        step_lr_cov = self.lr_cov / 100.0
      elif self.steps < 200 * self.warmup_factor:
        step_lr_cov = self.lr_cov / 10.0
      else:
        step_lr_cov = self.lr_cov
      lr1 = step_lr_cov

    beta2 = self.beta2
    log_scaling = np.log(scaling)
    log_scaling_mom = self.log_scaling[m] - log_scaling
    self.log_scaling[m] = log_scaling

    total_dim = len(G.shape)
    for idx, dim in enumerate(G.shape):
      name = f"{m}_dim-{total_dim - idx}"
      if total_dim > 1:
        assert dim > 1
      if dim == 1:
        assert total_dim == 1

      if self.using_matrix_norm:
        self.precond_m_B_blocks[name].mul_(
            beta2 * np.exp(log_scaling_mom)).add_(
                res[name], alpha=(1.0 - beta2)
            )  # beta2: alpha_1 in the paper  (riemannian momentum for K)
      else:
        self.precond_m_B_blocks[name].mul_(
            beta2 * np.exp(log_scaling_mom)).add_(
                res[name], alpha=1.0
            )  # beta2: alpha_1 in the paper  (riemannian momentum for K)

      norm_B = 1.0 / scaling
      if self.using_matrix_norm:
        norm_ = torch.max(torch.abs(self.precond_m_B_blocks[name]))
        #norm_ = torch.norm(self.precond_m_B_blocks[name])
        norm_B = torch.max(torch.tensor([norm_B, norm_]))

      self.precond_B_blocks[name].add_(
          (self.precond_B_blocks[name] @ self.precond_m_B_blocks[name]),
          alpha=-lr1 / norm_B
      )  # lr1:beta_1 in the paper  (first-order truncation for the expm)

      scaling_B = self.get_scaling(self.precond_B_blocks[name])
      tmp_B = self.precond_B_blocks[name] / scaling_B
      self.precond_BBt_blocks[name] = tmp_B @ (tmp_B.t())
      assert torch.isfinite(self.precond_BBt_blocks[name]).all()
      self.scaling[m] *= scaling_B**2

  def _group_param_grad(
      self,
      block: nn.Parameter,
      key: str,
      cast_dtype: torch.dtype = torch.float32,
  ) -> torch.Tensor:
    assert len(block) == 1
    W = torch.squeeze(block[0].grad)
    assert W is not None
    # if W.dim() > 2 and self.steps == 0:
    #   logging.info(f"{W.shape}, {key}")

    ################################
    # W = W.view(len(W), -1)
    # W = torch.squeeze(W)
    ################################
    assert torch.isfinite(W).all()

    return W.to(dtype=cast_dtype)

  def _update_natural_grad(self,
                           m: nn.Module,
                           block: List[nn.Parameter],
                           p_grad: torch.Tensor,
                           damping: float):
    """
        :param m:  the layer
        :param p_grad: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
    total_dim = p_grad.dim()
    name1 = f"{m}_dim-1"
    name2 = f"{m}_dim-2"
    name3 = f"{m}_dim-3"
    name4 = f"{m}_dim-4"
    if total_dim == 1:
      v = self.precond_BBt_blocks[name1] @ p_grad
    elif total_dim == 2:
      v = self.precond_BBt_blocks[name2] @ p_grad @ self.precond_BBt_blocks[
          name1]
    elif total_dim == 3:
      v = torch.einsum('pi,ijk->pjk', self.precond_BBt_blocks[name3], p_grad)
      v = torch.einsum('pjk,jq->pqk', v, self.precond_BBt_blocks[name2])
      v = torch.einsum('pqk,km->pqm', v, self.precond_BBt_blocks[name1])
    elif total_dim == 4:
      v = torch.einsum('pi,ijkl->pjkl', self.precond_BBt_blocks[name4], p_grad)
      v = torch.einsum('pjkl,jq->pqkl', v, self.precond_BBt_blocks[name3])
      v = torch.einsum('pqkl,km->pqml', v, self.precond_BBt_blocks[name2])
      v = torch.einsum('pqml,lw->pqmw', v, self.precond_BBt_blocks[name1])
    else:
      raise ValueError(
          f"Only up to 4d tensor is supported, but got {total_dim}.")

    v = v * self.scaling[m]
    assert torch.isfinite(v).all()

    block[0].grad.data.copy_(v.view(block[0].grad.data.size()))

    return v

  def _step(self):
    for group in self.param_groups:
      weight_decay = group["weight_decay"]
      momentum = group["momentum"]

      if self.steps == 0:
        self.previous_lr = group["lr"]
      else:
        self.previous_lr = group["lr"]

      for p in group["params"]:
        p: nn.Parameter
        if p.grad is None:
          continue

        d_p = p.grad.data  # grad or natural_grad
        if weight_decay != 0:
          # add weight decay into momentum
          d_p.add_(p.data, alpha=weight_decay)
          # do not add weight decay into momentum
          #p.data.mul_(1 - group["lr"] * weight_decay)

        if momentum != 0:  # add momentum
          param_state = self.state[p]
          if "momentum_buffer" not in param_state:
            # note that this is float32
            #buf = param_state["momentum_buffer"] = torch.zeros_like(d_p)
            buf = param_state["momentum_buffer"] = torch.zeros_like(
                d_p.to(self.cast_dtype))
          else:
            buf = param_state["momentum_buffer"]
          buf.mul_(momentum).add_(d_p)  # add the standard momentum
          # self.record_info( buf, 'm_w' )
          d_p = buf

        assert torch.isfinite(d_p).all()
        p.data.add_(d_p, alpha=-group["lr"])  # perform a SGD-like update
        # self.record_info( p.data, 'w' )

  def get_scaling(self, A: torch.Tensor) -> float:
    return np.max([1.0, torch.sqrt(torch.max(torch.abs(A))).item()])

  @torch.no_grad()
  def step(self, closure=None):
    del closure
    damping = self.damping

    for key, block in self.params_list.items():
      p_grad = self._group_param_grad(block, key, cast_dtype=self.cast_dtype)

      if self.steps == 0:
        self.log_scaling[key] = 0.0
        self.scaling[key] = 1.0
        total_dim = len(p_grad.shape)
        for idx, dim in enumerate(p_grad.shape):
          name = f"{key}_dim-{total_dim - idx}"

          self.precond_B_blocks[name] = torch.diag(p_grad.new(dim).fill_(1))
          self.precond_m_B_blocks[name] = torch.zeros_like(
              self.precond_B_blocks[name])
          self.precond_BBt_blocks[name] = torch.ones_like(
              self.precond_B_blocks[name])

      if self.steps == self.next_step:
        factor = 1.0  # since grad is unscaled
        if self.batch_averaged:
          factor *= math.sqrt(self.batch_size)

        self._update_inv(key, factor * p_grad)  # inverse fim/hessian estimation

      self._update_natural_grad(key, block, p_grad, damping)

    if self.steps == self.next_step:
      diff = min(max(int(math.log(self.steps + 1, 4)), 1), self.T)
      self.next_step = diff + self.steps
      #logging.info(f'next step is {self.next_step} {diff} {self.steps}')

    self._step()
    self.steps += 1

    # if len(self.is_max_update)>0 and self.is_print:
    # logging.info('-----------------------------------------------')
    # info = [ (k, self.max_info[k]) for k in self.is_max_update ]
    # logging.info(info)
    # logging.info('++++++++++++++++++++++++++++++++++++++++++++++++')

    # if ('max_tr_g_Tg_tr_x_Tx' in self.is_max_update
    #     or 'max_tr_H_K_tr_H_C' in self.is_max_update):
    # info_ ={
    # 'max_tr_g_Tg_tr_x_Tx': self.max_info['max_tr_g_Tg_tr_x_Tx'],
    # 'max_tr_H_K_tr_H_C': self.max_info['max_tr_H_K_tr_H_C'],
    # }
    # logging.info(info_)
    # self.is_max_update = set()
