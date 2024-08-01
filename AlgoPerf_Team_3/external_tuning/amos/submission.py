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
import enum

import numpy as np




#### amos.py ####


ScalarOrSchedule = optax.ScalarOrSchedule
Shape = Tuple[int, ...]
ParamsFn = Callable[[Tuple[jax.tree_util.DictKey, ...], Shape], Any]


class ScaleByAmosState(NamedTuple):
  """State for the Amos learning schedules."""

  b: optax.Updates


def scale_by_amos(
    xi: float,
    eta_fn: ParamsFn,
    l2: float = 0.0,
    *,
    adaptive: bool = False,
) -> optax.GradientTransformationExtraArgs:
  """Implements the Amos schedules for learning-rate and weight-decay.

  Args:
    xi: Global learning-rate.
    eta_fn: Variable-specific scale factor. A function that maps a variable name
      and shape to the hyper-parameter 'eta', estimating the scale of entries.
    l2: Strength of (extra) L2 regularization. Defaults to 0.
    adaptive: bool. Whether to make `b` adaptive (i.e. depending on gradient).
      Defaults to False (i.e. `b` only depends on global learning rate, global
      step and dimension k). Can improve performance when the neural network has
      sparse components (Embedding, MoE, etc.), but requires slightly more
      memory to store `b` (i.e. `b` is of reduced shape.)

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    b = jax.tree_map(lambda y: jnp.zeros((), y.dtype), params)
    return ScaleByAmosState(b=b)

  def update_fn(updates, state, params, g2, **other_args):
    """Applies per-variable learning-rate schedules and weight-decay.

    It assumes `updates` to be *normalized* gradients, and requires an extra
    `g2` arg to be normalized gradient squares of redueced shape.

    Args:
      updates: A tree of *normalized* gradients.
      state: A `ScaleByAmosState` object.
      params: The current value of the parameters.
      g2: Extra normalized gradient squares, of redueced shape. The shape of
        `g2` is used to calculate the reduced dimension k.
      **other_args: Other args. Unused.

    Returns:
      An (updates, state) tuple.
    """
    del other_args  # Unused.

    xi2 = jnp.square(xi)
    flat_params, tree_def = jax.tree_util.tree_flatten_with_path(params)
    flat_b = tree_def.flatten_up_to(state.b)
    flat_g = tree_def.flatten_up_to(updates)
    flat_g2 = tree_def.flatten_up_to(g2)
    for i, (name, theta) in enumerate(flat_params):
      g = flat_g[i]
      g2 = flat_g2[i]

      sqrt_k = np.sqrt(np.prod(theta.shape) / np.prod(g2.shape))
      tau = calc_tau(sqrt_k)
      l2p = 1 + l2 * ((1 + 0.5 * tau) ** 2 - 1) ** 2

      b = flat_b[i]
      sqrt_1b = jnp.sqrt(1 + b)
      xi2_rsqrt1b = xi2 / (jnp.maximum(g2, sqrt_1b) if adaptive else sqrt_1b)
      flat_b[i] = b + xi2_rsqrt1b * ((g2 if adaptive else 1) / sqrt_k + b)

      l2q = 1 + l2 * b * b
      wd = -0.5 * xi2_rsqrt1b * (l2p / l2q * g2 + l2)
      bv = (2 * sqrt_1b / (1 + jnp.sqrt(1 + 2 / (2 + tau) * sqrt_1b))) ** 2 - 1
      l2r = (1 + l2 * b * bv) / (1 + l2 / l2p * l2q)


      eta = eta_fn(name, theta.shape)
      upd = (wd * theta - eta * xi * g) / (1 + eta * sqrt_1b * l2r * bv)
      flat_g[i] = upd

    updates = tree_def.unflatten(flat_g)
    b = tree_def.unflatten(flat_b)
    return updates, ScaleByAmosState(b=b)

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def reduce_to_shape(x: jax.typing.ArrayLike, shape: Shape) -> jnp.ndarray:
  """Reduce mean to a given shape."""
  x = jnp.asarray(x)
  assert np.broadcast_shapes(x.shape, shape) == x.shape
  axis = [j for j, k in enumerate(shape) if k == 1] if shape else None
  return jnp.mean(x, axis=axis, keepdims=axis is not None)


def get_reduced_shape(shape_fn, name, theta):
  """Returns reduced shape for a given variable."""
  shape = () if shape_fn is None else shape_fn(name, theta.shape)
  assert np.broadcast_shapes(theta.shape, shape) == theta.shape
  return shape


def update_nu(nu, beta2, grad2):
  """Updates running average of gradient squares."""
  return jax.tree_util.tree_map(lambda v, g2: beta2 * v + g2, nu, grad2)


def bias_correct_nu(nu, beta2, count):
  """Bias correction for `nu`."""
  nu_correct = (1 - beta2) / (1 - jnp.power(beta2, count))
  return jax.tree_util.tree_map(lambda v: nu_correct * v, nu)


def nu_clip(nu, max_scale, grad, grad2):
  """Adaptive clipping by running average of grad squares (stored in `nu`)."""
  r = jax.tree_util.tree_map(
      lambda v, g2: jnp.maximum(jnp.where(v > 0, g2 / (v * max_scale), 1), 1),
      nu,
      grad2,
  )
  grad = jax.tree_util.tree_map(lambda g, x: g * jax.lax.rsqrt(x), grad, r)
  grad2 = jax.tree_util.tree_map(lambda g2, x: g2 / x, grad2, r)
  return grad, grad2


def adam_with_amos(
    learning_rate: ScalarOrSchedule,
    eta_fn: ParamsFn,
    shape_fn: Optional[ParamsFn] = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    l2: float = 0.0,
    *,
    use_nu_clip: bool = True,
) -> optax.GradientTransformationExtraArgs:
  """Adam optimizer with Amos learning-rate and weight-decay schedules.

  Args:
    learning_rate: A float or callable for the global learning-rate. When it is
      callable, it takes step count as input and returns a float scalar. For
      Amos we usually set a constant learning-rate or linear-warmup-to-constant
      schecule. Decaying schedules will be handled in the algorithm for each
      variable. One can use the `calc_base_lr` function to calculate a constant
      for this arg.
    eta_fn: A function that maps a variable name and shape to the variable-
      specific hyper-parameter 'eta', indicating the expected scale of entries.
    shape_fn: A function that maps a variable to a reduced shape.
    beta1: Exponential rate for moving average of updates (i.e. momentum).
    beta2: Exponential rate for moving average of squared gradients.
    l2: Strength of (extra) L2 regularization. Defaults to 0.
    use_nu_clip: bool. Whether to clip gradients according to previous running
      average of gradient squares. Defaults to True.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    # Adam state init.
    mu = jax.tree_util.tree_map(jnp.zeros_like, params)  # First moment
    nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
    adam_state = optax.ScaleByAdamState(
        count=jnp.zeros((), jnp.int32), mu=mu, nu=nu
    )

    # Amos state init.
    b = jax.tree_util.tree_map_with_path(
        lambda x, y: jnp.zeros(get_reduced_shape(shape_fn, x, y), y.dtype),
        params,
    )
    amos_state = ScaleByAmosState(b=b)
    return adam_state, amos_state

  def update_fn(grad, state, params, **extra_args):
    del extra_args  # Unused.
    adam_state, amos_state = state

    count = adam_state.count
    nu = adam_state.nu
    grad2 = jax.tree_util.tree_map(jnp.square, grad)
    if use_nu_clip:
      grad, grad2 = nu_clip(
          bias_correct_nu(nu, beta2, count), 1 / (1 - beta2), grad, grad2
      )

    # Adam updates.
    mu = jax.tree_util.tree_map(
        lambda g, m: (1 - beta1) * g + beta1 * m, grad, adam_state.mu
    )
    nu = update_nu(nu, beta2, grad2)
    count = optax.safe_int32_increment(count)
    nu_hat = bias_correct_nu(nu, beta2, count)
    updates = jax.tree_util.tree_map(
        lambda m, v: m * jax.lax.rsqrt(jnp.maximum(v, jnp.finfo(v.dtype).tiny)),
        mu,
        nu_hat,
    )
    adam_state = optax.ScaleByAdamState(count=count, mu=mu, nu=nu)

    # Apply Amos schedules.
    xi = learning_rate(count) if callable(learning_rate) else learning_rate
    amos_update_fn = scale_by_amos(xi, eta_fn, l2, adaptive=True).update
    m2_correct = (1 + beta1) / (1 - beta1)
    g2 = jax.tree_util.tree_map(
        lambda m, v, b: m2_correct
        * reduce_to_shape(jnp.square(m), b.shape)
        / jnp.maximum(reduce_to_shape(v, b.shape), jnp.finfo(v.dtype).tiny),
        mu,
        nu_hat,
        amos_state.b,
    )
    updates, amos_state = amos_update_fn(updates, amos_state, params, g2=g2)
    return updates, (adam_state, amos_state)

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)




def reduced_amos(
    learning_rate: ScalarOrSchedule,
    eta_fn: ParamsFn,
    shape_fn: Optional[ParamsFn] = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    l2: float = 0.0,
    *,
    use_nu_clip: bool = True,
) -> optax.GradientTransformationExtraArgs:
  """Adam-like, simplified, reduced memory usage, with Amos schedules.

  Args:
    learning_rate: A float or callable for the global learning-rate. When it is
      callable, it takes step count as input and returns a float scalar. For
      Amos we usually set a constant learning-rate or linear-warmup-to-constant
      schecule. Decaying schedules will be handled in the algorithm for each
      variable. One can use the `calc_base_lr` function to calculate a constant
      for this arg.
    eta_fn: A function that maps a variable name and shape to the variable-
      specific hyper-parameter 'eta', indicating the expected scale of entries.
    shape_fn: A function that maps a variable to a reduced shape.
    beta1: Exponential rate for moving average of updates (i.e. momentum).
    beta2: Exponential rate for moving average of squared gradients.
    l2: Strength of (extra) L2 regularization. Defaults to 0.
    use_nu_clip: bool. Whether to clip gradients according to previous running
      average of gradient squares. Defaults to True.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    # Adam-like state init, `nu` of reduced shape.
    mu = jax.tree_util.tree_map(jnp.zeros_like, params) if beta1 > 0 else None
    nu = jax.tree_util.tree_map_with_path(
        lambda x, y: jnp.zeros(get_reduced_shape(shape_fn, x, y), y.dtype),
        params,
    )
    adam_state = optax.ScaleByAdamState(
        count=jnp.zeros((), jnp.int32), mu=mu, nu=nu
    )

    # Amos state init.
    b = jax.tree_util.tree_map(lambda y: jnp.zeros((), y.dtype), params)
    amos_state = ScaleByAmosState(b=b)
    return adam_state, amos_state

  def update_fn(grad, state, params, **extra_args):
    del extra_args  # Unused.
    adam_state, amos_state = state

    count = adam_state.count
    nu = adam_state.nu
    grad2 = jax.tree_util.tree_map(jnp.square, grad)
    if use_nu_clip:
      grad, grad2 = nu_clip(
          bias_correct_nu(nu, beta2, count), 1 / (1 - beta2), grad, grad2
      )

    # Updates with reduced shape.
    reduced_grad2 = jax.tree_util.tree_map(
        lambda g2, v: reduce_to_shape(g2, v.shape), grad2, nu
    )
    nu = update_nu(nu, beta2, reduced_grad2)
    count = optax.safe_int32_increment(count)
    nu_hat = jax.tree_util.tree_map(
        lambda v: jnp.maximum(v, jnp.finfo(v.dtype).tiny),
        bias_correct_nu(nu, beta2, count),
    )
    updates = jax.tree_util.tree_map(
        lambda g, v: g * jax.lax.rsqrt(v), grad, nu_hat
    )

    # Apply Amos schedules.
    xi = learning_rate(count) if callable(learning_rate) else learning_rate
    amos_update_fn = scale_by_amos(xi, eta_fn, l2).update
    g2 = jax.tree_util.tree_map(lambda g2, v: g2 / v, reduced_grad2, nu_hat)
    updates, amos_state = amos_update_fn(updates, amos_state, params, g2=g2)

    # Put momentum to the last.
    if beta1 > 0:
      updates = jax.tree_util.tree_map(
          lambda g, m: (1 - beta1) * g + beta1 * m, updates, adam_state.mu
      )
      mu = updates
    else:
      mu = None
    adam_state = optax.ScaleByAdamState(count=count, mu=mu, nu=nu)
    return updates, (adam_state, amos_state)

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def calc_tau(sqrt_k: float, b: float = float('inf')) -> float:
  """Extra time due to early-stage."""
  q1m = np.sqrt(1 - 1 / sqrt_k)
  if b == float('inf'):
    return q1m * np.log(sqrt_k * (1 + q1m) ** 2)

  ratio = (1 + q1m) / (np.sqrt(1 + b) + q1m)
  return q1m * np.log((1 + sqrt_k * b) * ratio**2)


def get_max_k(
    params, shape_fn: Optional[ParamsFn] = None, shape_attr: str = 'shape'
) -> int:
  """Returns the maximum reduced dimension of variables."""
  flat_params, _ = jax.tree_util.tree_flatten_with_path(params)
  max_k = 1
  for name, theta in flat_params:
    full_shape = getattr(theta, shape_attr)
    reduced_shape = () if shape_fn is None else shape_fn(name, full_shape)
    assert np.broadcast_shapes(full_shape, reduced_shape) == full_shape
    k = np.prod(full_shape) // np.prod(reduced_shape)
    max_k = max(k, max_k)
  return max_k


def calc_base_lr(max_k: int, converge_steps: float) -> float:
  """Calculates the base learning-rate from model size and training budget.

  Args:
    max_k: int. The largest reduced dimension of model variables.
    converge_steps: float. The expected number of training steps to converge.

  Returns:
    base_lr: float. The base learning-rate.
  """
  xi2 = 2 * (calc_tau(np.sqrt(max_k)) + np.sqrt(2) - 1) / converge_steps
  base_lr = np.sqrt(xi2)
  return base_lr

#### end of amos.py #####


#### Start of param types #####



# Define this so that if using pytree iteration utilities, can iterate over the
# model shapes pytree without iterating over the shape tuples.
class ShapeTuple:
  def __init__(self, shape_tuple):
    self.shape_tuple = shape_tuple

  def __repr__(self):
    return f'ShapeTuple({self.shape_tuple})'

  def __eq__(self, other):
    return self.shape_tuple == other.shape_tuple

  def __hash__(self):
    return hash(repr(self))


def param_shapes(params):
  return jax.tree_map(lambda x: ShapeTuple(x.shape), flax.core.unfreeze(params))


class ParameterType(enum.Enum):
  """Different types of neural network parameters."""
  WEIGHT = 0
  BIAS = 1
  CONV_WEIGHT = 2
  BATCH_NORM_SCALE = 3
  BATCH_NORM_BIAS = 4
  LAYER_NORM_SCALE = 5
  LAYER_NORM_BIAS = 6
  EMBEDDING = 7
  ATTENTION_Q = 8
  ATTENTION_K = 9
  ATTENTION_V = 10
  ATTENTION_OUT = 11
  ATTENTION_QKV = 12  # This is used for implementations that fuse QKV together.
  # We need to split this out because otherwise fused QKV models will have a
  # different number of biases.
  ATTENTION_BIAS = 13


def param_types(shapes, parent_name: str = '') -> Dict[str, ParameterType]:
  """Get the ParameterType of each parameter."""
  param_types_dict = {}
  for name, value in shapes.items():
    name = name.lower()
    print('lowered name = ', name)

    if isinstance(value, dict):
      param_types_dict[name] = param_types(
          value, parent_name=parent_name + '/' + name)
    else:
      if 'batchnorm' in parent_name or 'bn' in parent_name:
        if name == 'scale':
          param_types_dict[name] = ParameterType.BATCH_NORM_SCALE
        elif name == 'bias':
          param_types_dict[name] = ParameterType.BATCH_NORM_BIAS
        else:
          raise ValueError(
              f'Unrecognized batch norm parameter: {parent_name}/{name}.')
      elif 'layernorm' in parent_name or 'ln' in parent_name:
        if name == 'scale':
          param_types_dict[name] = ParameterType.LAYER_NORM_SCALE
        elif name == 'bias':
          param_types_dict[name] = ParameterType.LAYER_NORM_BIAS
        else:
          raise ValueError(
              f'Unrecognized layer norm parameter: {parent_name}/{name}.')
      elif 'conv' in parent_name:
        if 'bias' in name:
          param_types_dict[name] = ParameterType.BIAS
        else:
          param_types_dict[name] = ParameterType.CONV_WEIGHT
      # Note that this is exact equality, not contained in, because
      # flax.linen.Embed names the embedding parameter "embedding"
      # https://github.com/google/flax/blob/main/flax/linen/linear.py#L604.
      elif ('embedding' in name or
            ('embedding' in parent_name and name == 'kernel')):
        param_types_dict[name] = ParameterType.EMBEDDING
      elif 'attention' in parent_name:
        if 'key' in parent_name and name == 'kernel':
          param_types_dict[name] = ParameterType.ATTENTION_K
        elif 'query' in parent_name and name == 'kernel':
          param_types_dict[name] = ParameterType.ATTENTION_Q
        elif 'value' in parent_name and name == 'kernel':
          param_types_dict[name] = ParameterType.ATTENTION_V
        elif 'out' in parent_name and name == 'kernel':
          param_types_dict[name] = ParameterType.ATTENTION_OUT
        elif name == 'bias':
          param_types_dict[name] = ParameterType.ATTENTION_BIAS
        elif 'scale' in name:
          param_types_dict[name] = ParameterType.WEIGHT
        elif 'in_proj_weight' in name:
          param_types_dict[name] = ParameterType.ATTENTION_QKV
        else:
          raise ValueError(
              f'Unrecognized attention parameter: {parent_name}/{name}.')
      elif 'bias' in name:
        param_types_dict[name] = ParameterType.BIAS
      else:
        param_types_dict[name] = ParameterType.WEIGHT
  return param_types_dict


  #### End of param_types ####

#### Amos helper ####
"""Helper utilities for the Amos optimizer."""
import ast
import math
import operator as op
import re
from typing import Any, Dict, Tuple

from absl import logging
from jax.sharding import PartitionSpec  # pylint: disable=g-importing-member
import numpy

_BIN_OP_MAP = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}


def evaluate(s: str, shape: Shape):
  """Evaluate simple expression. Allow 'SHAPE' referring to variable shape."""

  def _evaluate(node):
    if node is None:
      return None

    if isinstance(node, ast.BinOp):
      left = _evaluate(node.left)
      right = _evaluate(node.right)
      return _BIN_OP_MAP[type(node.op)](left, right)

    if isinstance(node, ast.Call):
      func_name = node.func
      assert isinstance(func_name, ast.Name)
      func = getattr(math, func_name.id, None)
      if func is None:
        func = getattr(numpy, func_name.id)

      assert not node.keywords
      args = [_evaluate(x) for x in node.args]
      return func(*args)

    if isinstance(node, ast.Constant):
      return node.value

    if isinstance(node, ast.Name):
      assert node.id == 'SHAPE'
      return shape

    if isinstance(node, ast.Num):  # Python 3.7 compatibility
      return node.n

    if isinstance(node, ast.Index):  # Python 3.8 compatibility
      return _evaluate(node.value)

    if isinstance(node, ast.Slice):
      return slice(
          _evaluate(node.lower), _evaluate(node.upper), _evaluate(node.step))

    if isinstance(node, ast.Subscript):
      return _evaluate(node.value)[_evaluate(node.slice)]

    if isinstance(node, ast.Tuple):
      return tuple([_evaluate(x) for x in node.elts])

    if isinstance(node, ast.UnaryOp):
      assert isinstance(node.op, ast.USub)
      return -_evaluate(node.operand)  # pylint: disable=invalid-unary-operand-type

    raise TypeError(f'Cannot handle node type: {type(node).__name__}')

  node = ast.parse(s, mode='eval').body
  return _evaluate(node)


def params_fn_from_assign_map(assign_map: Dict[str, Any],
                              name_sep: str = '/',
                              eval_str_value: bool = False) -> ParamsFn:
    """Creates a params_fn from assign_map.

    A params_fn maps each variable name and shape to some value. The variable name
    is a tuple of str, and shape is a tuple of int. An assign_map is a sequence of
    rules, where each rule maps a regex of variable names to a value.

    Args:
        assign_map: A dictionary mapping 'regex' to 'value'. Given a variable name,
        the returned params_fn will find the first matching 'regex' and return the
        corresponding 'value'.
        name_sep: Join the the variable name (tuple of str) by this separator before
        regex matching. Defaults to '/'.
        eval_str_value: If True, value can be str of simple expressions, which will
        be evaluated.

    Returns:
        params_fn: A function that maps each variable name and shape to a value.
    """

    for regex, value in assign_map.items():
      if re.match(regex, name_str):
        logging.info('Matched rule (%s -> %s) to variable %s of shape %s.',
                     regex, value, name, shape)
        if eval_str_value and isinstance(value, str):
          return evaluate(value, shape)
        return value
    raise ValueError(f'No matching rule for variable {name} of shape {shape}.')

    return params_fn

### End of Amos helper ####

def instantiate_optimizer(
    b1: float = 0.9,
    b2: float = 0.999
) -> optax.GradientTransformation:
  """Rescale updates according to the NAdamP algorithm.

  Returns:
    An (init_fn, update_fn) tuple.
  """
  return None



def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
    """Creates optimizer state and a learning rate schedule."""
    del model_state
    del rng

    if hyperparameters.style == 'adam':
        amos_optimizer = adam_with_amos
    elif hyperparameters.style == 'reduced':
        amos_optimizer = reduced_amos
    else:
        raise ValueError('Unknown amos style: %s' % hyperparameters.style)

    print(amos_optimizer)
    unreplicated_params = jax_utils.unreplicate(model_params)

    model_param_shapes = param_shapes(unreplicated_params)
    model_param_types = param_types(model_param_shapes)

    flattened_param_shapes = flax.traverse_util.flatten_dict(model_param_shapes, sep='/')
    flattened_params_types = flax.traverse_util.flatten_dict(model_param_types, sep='/')

    print('shapes = ', flattened_param_shapes.keys())
    print('\n')

    print('types = ', flattened_params_types.keys())
    

    shape_rules = {
        ParameterType.WEIGHT: (),
        ParameterType.BIAS: (),
        ParameterType.CONV_WEIGHT: (),
        ParameterType.BATCH_NORM_SCALE: (),
        ParameterType.BATCH_NORM_BIAS: (),
        ParameterType.LAYER_NORM_SCALE: (),
        ParameterType.LAYER_NORM_BIAS: (),
        ParameterType.EMBEDDING: '(SHAPE[0], 1)',
        ParameterType.ATTENTION_Q: '(1, SHAPE[1], 1)',
        ParameterType.ATTENTION_K: '(1, SHAPE[1], 1)',
        ParameterType.ATTENTION_V: '(1, SHAPE[1], 1)',
        ParameterType.ATTENTION_OUT: (),
        ParameterType.ATTENTION_QKV: '(1, SHAPE[1], 1)',
        ParameterType.ATTENTION_BIAS: (),
    }

    def shape_fn(name, shape):
        name_str = '/'.join([str(x.key) for x in name]).lower()
        print(name_str)
        print(flattened_params_types[name_str])

        ret = shape_rules[flattened_params_types[name_str]]
        if isinstance(ret, str):
            ret = evaluate(ret, shape)
        return ret

    max_k = get_max_k(flattened_param_shapes, shape_fn, shape_attr='shape_tuple')
    warmup = hyperparameters.warmup * workload.step_hint
    effective_steps = workload.step_hint - warmup * 2 / 3
    converge_factor = hyperparameters.converge
    converge_steps = converge_factor * effective_steps
    base_lr = calc_base_lr(max_k, converge_factor * effective_steps)
    lr_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup,
        transition_begin=0,
    )

    eta_overrides = hyperparameters.eta_overrides

    eta_rules = {
        ParameterType.WEIGHT: 'sqrt(2/SHAPE[0])',
        ParameterType.BIAS: 0.5,
        ParameterType.CONV_WEIGHT: 'sqrt(1/prod(SHAPE[:-1]))',
        ParameterType.BATCH_NORM_SCALE: 1.0,
        ParameterType.BATCH_NORM_BIAS: 0.5,
        ParameterType.LAYER_NORM_SCALE: 1.0,
        ParameterType.LAYER_NORM_BIAS: 0.5,
        ParameterType.EMBEDDING:1.0,
        ParameterType.ATTENTION_Q: 'sqrt(1/SHAPE[0])',
        ParameterType.ATTENTION_K: 'sqrt(1/SHAPE[0])',
        ParameterType.ATTENTION_V: 'sqrt(1/SHAPE[0])',
        ParameterType.ATTENTION_OUT: 'sqrt(1/prod(SHAPE[:-1]))',
        ParameterType.ATTENTION_QKV: 'sqrt(1/SHAPE[0])',
        ParameterType.ATTENTION_BIAS: 0.5,
    }

    if eta_overrides != -1:
        eta_rules[ParameterType.WEIGHT] = eta_overrides
        eta_rules[ParameterType.BIAS] = eta_overrides
        eta_rules[ParameterType.EMBEDDING] = eta_overrides

    def eta_fn(name, shape):
        name_str = '/'.join([str(x.key) for x in name]).lower()
        ret = eta_rules[flattened_params_types[name_str]]
        if isinstance(ret, str):
            ret = evaluate(ret, shape)
        logging.info('eta %s: %s', name_str, ret)
        return ret

    l2 = hyperparameters.l2

    def global_clip_amos(learning_rate):
        return optax.chain(
            optax.clip_by_global_norm(1),
            amos_optimizer(
                learning_rate=learning_rate,
                eta_fn=eta_fn,
                shape_fn=shape_fn,
                beta1=hyperparameters.beta1,
                beta2=hyperparameters.beta2,
                l2=l2,
            ),
        )

    params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)

    opt_init_fn, opt_update_fn = global_clip_amos(
        learning_rate=lr_schedule
    )
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
  if global_step < 100 and workload.metrics_logger is not None:
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