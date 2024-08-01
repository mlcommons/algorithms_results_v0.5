import functools
from typing import Any, Callable, NamedTuple, Optional, Union
from jax.scipy.linalg import cho_factor, cho_solve


import chex
import jax
import math
from jax import lax
import numpy as np
import optax
import jax.numpy as jnp

# from reference_algorithms.paper_baselines.shampoo.jax.distributed_shampoo import matrix_inverse_pth_root,mat_power,power_iteration
from submissions.submissions_algorithms_v0_5.AlgoPerf_Team_16.external_tuning.caspr_adaptive.distributed_shampoo import matrix_inverse_pth_root,mat_power,power_iteration


# pylint:disable=no-value-for-parameter


ScalarOrSchedule = Union[float, optax.Schedule]
class TraceState(NamedTuple):
        """Holds an aggregation of past updates."""
        trace: optax.Params




MaskOrFn = Optional[Union[Any, Callable[[optax.Params], Any]]]




class ScaleByCasprState(NamedTuple):
        """State for the Adam algorithm."""
        count: chex.Array  # shape=(), dtype=jnp.int32.
        mu: optax.Updates
        nu: optax.Updates
        stats: optax.Updates
        preconds: optax.Updates
        prev_stats: optax.Updates


class CasprLRPair(NamedTuple):
        L: chex.Array
        R: chex.Array

def update_moment(updates, moments, decay, order):
        """Compute the exponential moving average of the `order`-th moment."""
        w1,w2 = (1-decay) if decay!=1.0 else 1.0, decay
        return jax.tree_util.tree_map(
            lambda g, t: w1 * (g ** order) + w2 * t, updates, moments)




def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
        m = -1 if flip_sign else 1
        if callable(learning_rate):
                return optax.scale_by_schedule(lambda count: m * learning_rate(count))
        return optax.scale(m * learning_rate)



@functools.partial(jax.jit, inline=True)
def bias_correction(moment, decay, count):
        """Performs bias correction. It becomes a no-op as count goes to infinity."""
        # The conversion to the data type of the moment ensures that bfloat16 remains
        # bfloat16 in the optimizer state. This conversion has to be done after
        # `bias_correction_` is calculated as calculating `decay**count` in low
        # precision can result in it being rounded to 1 and subsequently a
        # "division by zero" error.
        bias_correction_ = 1 - decay**count


        # Perform division in the original precision.
        return jax.tree_util.tree_map(
                lambda t: t / bias_correction_.astype(t.dtype), moment)


def abs_sq(x: chex.Array) -> chex.Array:
        """Returns the squared norm of a (maybe complex) array.


        For real `x`, JAX generates the same HLO from this, `jnp.square(x)`, `x * x`,
        or `x**2`.


        Args:
        x: a (maybe complex) array.


        Returns:
        The squared norm of `x`.
        """
        if not isinstance(x, (np.ndarray, jnp.ndarray)):
                raise ValueError(f"`abs_sq` accepts only NDarrays, got: {x}.")
        return (x.conj() * x).real


def update_moment_per_elem_norm(updates, moments, decay, order):
        """Compute the EMA of the `order`-th moment of the element-wise norm."""
        def orderth_norm(g):
                if jnp.isrealobj(g):
                        return g ** order
                else:
                        half_order = order / 2
                        # JAX generates different HLO for int and float `order`
                        if half_order.is_integer():
                                half_order = int(half_order)
                        return abs_sq(g) ** half_order
        w1,w2 = (1-decay) if decay!=1.0 else 1.0, decay
        return jax.tree_util.tree_map(
                lambda g, t: w1 * orderth_norm(g) + w2 * t, updates, moments)




def get_merged_shape(param_shape):
        assert len(param_shape)<=4
        if len(param_shape)==4:
                new_shape = (param_shape[0]*param_shape[1]*param_shape[2], param_shape[3])
                return new_shape
        elif len(param_shape)==3:
                new_shape = (param_shape[0], param_shape[1]*param_shape[2]) if param_shape[1]<param_shape[0] else (param_shape[0]*param_shape[1],param_shape[2])
                return new_shape
        else:
                return param_shape

def get_blocked_shape(merged_shape,block_size):
        assert len(merged_shape)==2
        d1, d2 = merged_shape
        return (math.ceil(d1/block_size),math.ceil(d2/block_size),block_size,block_size)

def get_padded_matrix(padding_start,block_size=1024):
      ix = (jnp.arange(block_size)<padding_start)
      return (jnp.eye(block_size)*ix[jnp.newaxis,:])*ix[:,jnp.newaxis]




def regularized_stat(stat_L,update,block_size,matrix_epsilon):
    get_padded_matrix_vmap = jax.vmap(functools.partial(get_padded_matrix,block_size=block_size),in_axes=0)
    mgd_shape = get_merged_shape(update.shape)
    print("mgd_shape",mgd_shape)
    blkd_shape = get_blocked_shape(mgd_shape,block_size)
    g1,g2,_,_ = blkd_shape
    paddings = get_paddings(CasprLRPair(L=stat_L,R=optax.MaskedNode()), update, block_size)

    if mgd_shape[0]<=block_size:
      #this is an unpadded stat
      # print('id_L shape',id_L.shape)
      id_L = jnp.zeros((g1,g2,mgd_shape[0],mgd_shape[0]))
      id_L = id_L.at[:,:].set(jnp.eye(mgd_shape[0],mgd_shape[0]))
      print('id_L shape after',id_L.shape)
    else:
      id_L = get_padded_matrix_vmap(paddings.L.reshape(-1,1))
      id_L = id_L.reshape((blkd_shape[0],blkd_shape[1],block_size,block_size))
    g1,g2 = stat_L.shape[0],stat_L.shape[1]
    trval_L = (jnp.einsum("ijkk->ij",stat_L)/stat_L.shape[-1]+1e-30).reshape(-1)[:,jnp.newaxis,jnp.newaxis]
    max_ev_L = (jax.vmap(power_iteration)(stat_L.reshape(-1,stat_L.shape[-2],stat_L.shape[-1])/trval_L)[1]).reshape(g1,g2,1,1)
    print(update,stat_L,matrix_epsilon,max_ev_L,id_L)
    stat_L = stat_L+(matrix_epsilon*trval_L.reshape(g1,g2,1,1))*max_ev_L*id_L
    return stat_L



class UpdateStats(NamedTuple):
    stat: CasprLRPair
    prev_L: Any
    coeff: Any

def update_stats(L,R,prev_L,precond_L,grad,block_size,b2,fresh_preconds,matrix_epsilon):
    print("L R prev_L precond_L grad", L, R, prev_L, precond_L, grad)
    #TODO: update statistics once every few steps
    #  L, R = s.L, s.R

    mgd_shape = get_merged_shape(grad.shape)
    if len(mgd_shape)<=1 or sum([ dim>40000 for dim in mgd_shape]):
        return UpdateStats(stat=CasprLRPair(L,R),prev_L=optax.MaskedNode(),coeff=optax.MaskedNode())
    print(mgd_shape)
    grad = grad.reshape(mgd_shape)

    #let block_size be 1000
    # a) stat of type (1,1,500,500) or (1,1,1000,1000)
    # no left/right padding or reshapeing of grad needed
    # b) stat of type (2,1,1000,1000) or (2,2,1000,1000)
    # padding and reshaping of grad needed
    # c) stat of type (1,3,500,500)
    # padding of grad not needed on the left, but reshaping needed
    tr = lambda x,y: jnp.sum(jnp.sum(x*y,axis=-1),axis=-1)

    mm = lambda x,y: jnp.einsum('ijkl,ijlm->ijkm',x,y)
    if b2==1.0:
        w1, w2 = 1.0, 1.0
    else:
        w1, w2 = b2, 1.0-b2

    if mgd_shape[0]<=block_size and mgd_shape[1]<=block_size:
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1==1 and g2==1
      # jax.debug.print("tr precond_L.precondL.L {x}",x = jnp.trace(precond_L@precond_L@L)/L.shape[-1])

      coeff = jax.lax.cond(fresh_preconds,
              lambda: jnp.clip(tr(precond_L,
                                  regularized_stat(prev_L,
                                                   grad,
                                                   block_size,
                                                   matrix_epsilon)@precond_L)[:,:,jnp.newaxis,jnp.newaxis],0,None)/mgd_shape[0],
              lambda: jnp.ones((g1,g2,1,1)))
      # jax.debug.print((precond_L))
      prev_L = jax.lax.cond(fresh_preconds,lambda: L,lambda: prev_L)
      L = w1*L + w2*grad@grad.T
      precond_grad = (grad.T@precond_L)
      R = w1*coeff*R +  w2*jnp.einsum("ijkl,ijnl->ijkn",precond_grad,precond_grad)/mgd_shape[0]
      # jax.debug.print("residual {x}",x=jnp.trace((jnp.einsum("ijkl,ijnl->ijkn",precond_grad,precond_grad)/mgd_shape[0])[0,0]))

    if mgd_shape[0]>block_size and mgd_shape[1]<=block_size:
      # L will look as g1,1,block_size,block_size and R will look as g1,1,mgd_shape[1],mgd_shape[1]
      # grad is padded to the left
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1!=1 and g2==1
      M = np.zeros((g1,g2))
      M[-1,:] = mgd_shape[0]%block_size  if mgd_shape[0]%block_size!=0 else block_size
      M[:-1,:] = block_size
      coeff = jax.lax.cond(fresh_preconds,
                                        lambda: jnp.clip(tr(precond_L, regularized_stat(prev_L,
                                                  grad,
                                                   block_size,
                                                   matrix_epsilon)@precond_L)[:,:,jnp.newaxis,jnp.newaxis],0,None)/M[:,:,np.newaxis,np.newaxis],
                                        lambda: jnp.ones((g1,g2,1,1)))
      prev_L = jax.lax.cond(fresh_preconds,lambda: L,lambda: prev_L)
      grad = jnp.pad(grad,((0,(-mgd_shape[0])%block_size),(0,0)),mode='constant')
      grad = grad.reshape(g1,block_size,g2,mgd_shape[1])
      grad = grad.transpose((0,2,1,3))
      L = w1*L + w2*jnp.einsum("ijkl,ijnl->ijkn",grad,grad)
      precond_grad = jnp.einsum("ijkl,ijln->ijnk",precond_L,grad)
      R = w1*coeff*R + w2*jnp.einsum("ijkl,ijnl->ijkn",precond_grad,precond_grad)/M[:,:,np.newaxis,np.newaxis]
    if mgd_shape[0]<=block_size and mgd_shape[1]>block_size:
      # L will look as 1,g2,mgd_shape[0],mgd_shape[0] and R will look as 1,g2,block_size,block_size
      # grad is padded to the right
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1==1 and g2!=1
      M = np.zeros((g1,g2))
      M[-1,:] = mgd_shape[0]%block_size  if mgd_shape[0]%block_size!=0 else block_size
      M[:-1,:] = block_size
      coeff = jax.lax.cond(fresh_preconds,
                                        lambda: jnp.clip(tr(precond_L, regularized_stat(prev_L,
                                                  grad,
                                                   block_size,
                                                   matrix_epsilon)@precond_L)[:,:,jnp.newaxis,jnp.newaxis],0,None)/mgd_shape[0],
                                        lambda: jnp.ones((g1,g2,1,1)))
      prev_L = jax.lax.cond(fresh_preconds,lambda: L,lambda: prev_L)
      grad = jnp.pad(grad,((0,0),(0,(-mgd_shape[1])%block_size)),mode='constant')
      grad = grad.reshape(g1,mgd_shape[0],g2,block_size)
      grad = grad.transpose((0,2,1,3))
      L = w1*L + w2*jnp.einsum("ijkl,ijnl->ijkn",grad,grad)
      precond_grad = jnp.einsum("ijkl,ijln->ijnk",precond_L,grad)
      R = w1*coeff*R + w2*jnp.einsum("ijkl,ijnl->ijkn",precond_grad,precond_grad)/mgd_shape[0]
    if mgd_shape[0]>block_size and mgd_shape[1]>block_size:
      #L and R will look like g1,g2,block_size,block_size
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1!=1 and g2!=1
      M = np.zeros((g1,g2))
      M[-1,:] = mgd_shape[0]%block_size  if mgd_shape[0]%block_size!=0 else block_size
      M[:-1,:] = block_size
      coeff = jax.lax.cond(fresh_preconds,
                                        lambda: jnp.clip(tr(precond_L, regularized_stat(prev_L,
                                                  grad,
                                                   block_size,
                                                   matrix_epsilon)@precond_L)[:,:,jnp.newaxis,jnp.newaxis],0,None)/M[:,:,np.newaxis,np.newaxis],
                                        lambda: jnp.ones((g1,g2,1,1)))
      prev_L = jax.lax.cond(fresh_preconds,lambda: L,lambda: prev_L)
      grad = jnp.pad(grad,((0,(-mgd_shape[0])%block_size),(0,(-mgd_shape[1])%block_size)),mode='constant')
      grad = grad.reshape(g1,mgd_shape[0],g2,block_size)
      grad = grad.transpose((0,2,1,3))
      L = w1*L + w2*jnp.einsum("ijkl,ijnl->ijkn",grad,grad)
      precond_grad = jnp.einsum("ijkl,ijln->ijnk",precond_L,grad)
      R = w1*coeff*R + w2*jnp.einsum("ijkl,ijnl->ijkn",precond_grad,precond_grad)/M[:,:,np.newaxis,np.newaxis]

    return UpdateStats(stat=CasprLRPair(L,R),prev_L=prev_L,coeff=coeff)


def eigh_inverse(stat,padding,exponent=2,epsilon=1e-6,relative_epsilon=True):
    # _,max_ev = power_iteration(stat)
    # max_ev = jnp.linalg.norm(stat,ord=1)
    # max_ev = jnp.trace(stat)/stat.shape[0]
    ix = (jnp.arange(stat.shape[0])<padding)
    stat = (stat*ix[:,jnp.newaxis])*ix[jnp.newaxis,:]
    identity = jnp.array((np.eye(stat.shape[0])*ix[:,np.newaxis])*ix[np.newaxis,:],dtype=stat.dtype)
    scale = (1e-30+jnp.trace(stat)/padding)
    stat = stat/scale

    _,max_ev = power_iteration(stat)


    if relative_epsilon:
        epsilon = jnp.maximum(epsilon*max_ev,1e-20)
    reg_stat = stat+identity*epsilon
    eigvals,eigvecs = jnp.linalg.eigh(reg_stat)

    mm = functools.partial(jnp.matmul,precision=lax.Precision.HIGHEST)
    inv_eigvals = (jnp.maximum(eigvals, epsilon)**(-1./exponent))

    inv_pth_reg_stat = mm(mm(eigvecs,jnp.diag(inv_eigvals)),eigvecs.T)
    inv_pth_reg_stat = (inv_pth_reg_stat*ix[:,jnp.newaxis])*ix[jnp.newaxis,:]
    error = jnp.max(jnp.abs(mat_power(inv_pth_reg_stat,p=exponent)@reg_stat - identity))
    # error = 1e-7
    return inv_pth_reg_stat*(scale**(-1/exponent)), error


def coupled_newton_inverse(stat,padding,exponent=2,epsilon=1e-6,relative_epsilon=True):
    scale = (jnp.trace(stat)/stat.shape[0]+1e-30)
    stat = stat/scale
    # ix = (jnp.arange(stat.shape[0])<padding)
    # stat = stat*(ix[:,jnp.newaxis])*ix[jnp.newaxis,:]
    inv_pth_root,metrics = matrix_inverse_pth_root(stat,exponent,ridge_epsilon=epsilon,error_tolerance=1e-6,
                            relative_matrix_epsilon=relative_epsilon,padding_start=padding)
    error = metrics.inverse_pth_root_errors

    # jax.debug.print("tr value {x}", x= jnp.sum(inv_pth_root*(stat@inv_pth_root))/padding)
    # inv_pth_root = inv_pth_root *(ix[:,jnp.newaxis])*ix[jnp.newaxis,:]

    return inv_pth_root*(scale**(-1/exponent)),error

def cholesky_inverse(stat,exponent=4,epsilon=1e-6,relative_epsilon=True):
        #exponent is not going to be used.
        # _,max_ev = power_iteration(stat)
        # max_ev = jnp.linalg.norm(stat,ord=1)
        max_ev = jnp.trace(stat)/stat.shape[0]
        # jnp.linalg.norm(stat,ord=2)
        if relative_epsilon:
                epsilon = jnp.maximum(epsilon*max_ev,1e-16)
        reg_stat = stat+jnp.eye(stat.shape[0])*epsilon
        # reg_stat = stat
        c, lower = cho_factor(reg_stat)


        # Solve the linear system using the Cholesky factorization
        val = cho_solve((c, lower), jnp.eye(reg_stat.shape[0]))
        mm = functools.partial(jnp.matmul,precision=lax.Precision.HIGHEST)
        error = jnp.max(jnp.abs(mm(val,reg_stat) - jnp.eye(reg_stat.shape[0])))
        return val, error


def batch_stats(x, num_devices):
        """Batch `x` so that so that leading axis is num_devices."""
        n = len(x)
        b = int(n / num_devices)
        return x.reshape(num_devices,b,x.shape[-2],x.shape[-1])

def batch_paddings(x, num_devices):
    """Batch `x` so that so that leading axis is num_devices."""
    n = len(x)
    b = int(n / num_devices)
    return x.reshape(num_devices,b,1)



def unbatch_stats(x):
        return x.reshape(-1,x.shape[-2],x.shape[-1])


def unbatch_errors(x):
        return x.reshape(-1)



def get_inverses(stats,paddings,exponent,epsilon,
                                     block_size,relative_epsilon,inverse_type,error_tolerance,batch_axis_name=None):

      if inverse_type=='eigh':
              inverse_fn = eigh_inverse
      elif inverse_type=='cholesky':
              inverse_fn = cholesky_inverse
      elif inverse_type=='coupled newton':
              inverse_fn = coupled_newton_inverse
      elif inverse_type=='rsvd':
              raise NotImplemented
      #  stats = jnp.stack([L,R],axis=0)
      assert len(stats.shape)==3
      g = stats.shape[0]
      stats_flat = stats
      paddings_flat = paddings
      if batch_axis_name:
              num_devices = lax.psum(1, batch_axis_name)
      else:
              num_devices = 1
      num_statistics = g
      #distribute inverse operations to multiple devices
      if batch_axis_name:
              # Pad statistics and exponents to next multiple of num_devices.
              to_pad = (-num_statistics) % num_devices
              pad_stats = jnp.zeros((to_pad,block_size,block_size))
              pad_paddings = jnp.ones((to_pad,1))*block_size
              pad_stats = pad_stats.at[:].set(jnp.eye(block_size, dtype=stats.dtype))
              stats_flat = jnp.concatenate([stats_flat, pad_stats], axis=0)
              paddings_flat = jnp.concatenate([paddings_flat, pad_paddings], axis=0)
              stats_flat_batched = batch_stats(stats_flat, num_devices)
              paddings_flat_batched = batch_paddings(paddings_flat, num_devices)
              current_replica = lax.axis_index(batch_axis_name)
              _matrix_inverse_pth_root_vmap = jax.vmap(functools.partial(inverse_fn,exponent=exponent,epsilon=epsilon,relative_epsilon=relative_epsilon))
              preconds_flat_batched, errors_flat_batched = _matrix_inverse_pth_root_vmap(
                      stats_flat_batched[current_replica], paddings_flat_batched[current_replica]
              )
              preconds_flat_batched = jax.lax.all_gather(preconds_flat_batched, batch_axis_name)
              print('to_pad', to_pad, 'errors_flat_batched',errors_flat_batched.shape,'preconds_flat_batched',preconds_flat_batched.shape)
              errors_flat_batched = jax.lax.all_gather(errors_flat_batched, batch_axis_name)
              print('to_pad', to_pad, 'errors_flat_batched',errors_flat_batched.shape,'preconds_flat_batched',preconds_flat_batched.shape)
              if to_pad!=0:
                      preconds_flat = unbatch_stats(preconds_flat_batched)[:-to_pad]
                      errors_flat = unbatch_errors(errors_flat_batched)[:-to_pad]
              else:
                      preconds_flat = unbatch_stats(preconds_flat_batched)
                      errors_flat = unbatch_errors(errors_flat_batched)
      else:
              preconds_flat, errors_flat = jax.vmap(functools.partial(inverse_fn,exponent=exponent,epsilon=epsilon,relative_epsilon=relative_epsilon))(stats_flat)
      preconds = preconds_flat.reshape(g,stats.shape[-2],stats.shape[-1])
      errors = errors_flat[:,jnp.newaxis,jnp.newaxis]
      errors = jnp.where(jnp.isnan(errors),jnp.ones_like(errors)*(error_tolerance+1.0),errors)
      # jax.debug.print('errors: {x}', x=errors)

      # print('preconds',preconds.shape)
      return preconds,errors




def split_array(arr, sizes):
        # Ensure the sum of sizes equals the first dimension of the array
        assert arr.shape[0] == sum(sizes), "The sum of sizes must equal the first dimension of the array"

        # Compute the cumulative sum of sizes to get split indices
        # We use [:-1] to exclude the last element, as split indices should not include the total length
        split_indices = np.cumsum(np.array(sizes))[:-1]

        # Split the array at the computed indices
        split_arrays = jnp.split(arr, split_indices)

        return split_arrays

def update_preconds_model(stats,preconds,paddings,mu,
                                        exponent,
                                        matrix_epsilon,
                                        block_size,
                                        relative_epsilon,
                                        inverse_type,
                                        error_tolerance,
                                        batch_axis_name):
      #TODO: do the following only when precondition is true.

        stats_flat,tree_def = jax.tree_util.tree_flatten(stats)
        paddings_flat,_ = jax.tree_util.tree_flatten(paddings)

        def pad_stat(stat):
          assert len(stat.shape)==4
          return jnp.pad(stat,((0,0),(0,0),(0,block_size-stat.shape[-2]),(0,block_size-stat.shape[-1])),mode='constant')

        #   assert not optax.MaskedNode() in stats_flat
        orig_shapes = []
        new_stats_flat = []
        new_paddings_flat = []
        for stat_flat,padding_flat in zip(stats_flat,paddings_flat):
                orig_shapes.append(stat_flat.shape)
                print(stat_flat.shape)
                new_stats_flat.append(pad_stat(stat_flat).reshape(-1,block_size,block_size))
                print(padding_flat)
                new_paddings_flat.append(padding_flat.reshape(-1,1))
        stats_flat = new_stats_flat
        paddings_flat = new_paddings_flat
        stats_flat = jnp.concatenate(stats_flat,axis=0)
        paddings_flat = jnp.concatenate(paddings_flat,axis=0)
        preconds_flat,errors_flat = get_inverses(stats_flat,paddings_flat,exponent,matrix_epsilon,
                                        block_size,relative_epsilon,inverse_type,error_tolerance,batch_axis_name)
        # jax.debug.print("paddings_flat {x}",x=paddings_flat)
        print("orig_shapes ",orig_shapes)
        #unwrapping preconds_flat
        split_sizes = ([ orig_shape[0]*orig_shape[1] for orig_shape in orig_shapes ])
        print("split_sizes",split_sizes)
        print(preconds_flat.shape,np.sum(split_sizes))
        errors_flat = split_array(errors_flat,split_sizes)
        preconds_flat = split_array(preconds_flat,split_sizes)
        errors_flat = [ error_flat.reshape(orig_shape[:2]) for error_flat,orig_shape in zip(errors_flat,orig_shapes)]
        preconds_flat = [ precond_flat[:,:orig_shape[-2],:orig_shape[-1]].reshape(orig_shape) for precond_flat,orig_shape in zip(preconds_flat,orig_shapes)]
        errors = jax.tree_util.tree_unflatten(tree_def,errors_flat)
        new_preconds = jax.tree_util.tree_unflatten(tree_def,preconds_flat)
        # jax.debug.print("errors {errors}",errors=errors)
        new_preconds = jax.tree_util.tree_map(lambda p,op,e: jnp.where(e[:,:,jnp.newaxis,jnp.newaxis]>error_tolerance,op,p),new_preconds,preconds,errors)

        return new_preconds




def caspr_update_fn(precond,momentum,adam_update,block_size):
    mgd_shape = get_merged_shape(momentum.shape)
    if len(mgd_shape)<=1 or sum([ dim>40000 for dim in mgd_shape]):
        return adam_update
    orig_shape= momentum.shape
    momentum = momentum.reshape(mgd_shape[0],mgd_shape[1])


    tr = lambda x,y: jnp.sum(jnp.sum(x*y,axis=-1),axis=-1)

    mm = lambda x,y: jnp.einsum('ijkl,ijlm->ijkm',x,y)

    #reshaping momentum
    if mgd_shape[0]<=block_size and mgd_shape[1]<=block_size:
      momentum = momentum[jnp.newaxis,jnp.newaxis,:,:]
    if mgd_shape[0]>block_size and mgd_shape[1]<=block_size:
      # L will look as g1,1,block_size,block_size and R will look as g1,1,mgd_shape[1],mgd_shape[1]
      # grad is padded to the left
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1!=1 and g2==1
      momentum = jnp.pad(momentum,((0,(-mgd_shape[0])%block_size),(0,0)),mode='constant')
      momentum = momentum.reshape(g1,block_size,g2,mgd_shape[1])
      momentum = momentum.transpose((0,2,1,3))
    if mgd_shape[0]<=block_size and mgd_shape[1]>block_size:
      # L will look as 1,g2,mgd_shape[0],mgd_shape[0] and R will look as 1,g2,block_size,block_size
      # grad is padded to the right
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1==1 and g2!=1
      momentum = jnp.pad(momentum,((0,0),(0,(-mgd_shape[1])%block_size)),mode='constant')
      momentum = momentum.reshape(g1,mgd_shape[0],g2,block_size)
      momentum = momentum.transpose((0,2,1,3))
    if mgd_shape[0]>block_size and mgd_shape[1]>block_size:
      #L and R will look like g1,g2,block_size,block_size
      blkd_shape = get_blocked_shape(mgd_shape,block_size)
      g1,g2,_,_ = blkd_shape
      assert g1!=1 and g2!=1
      momentum = jnp.pad(momentum,((0,(-mgd_shape[0])%block_size),(0,(-mgd_shape[1])%block_size)),mode='constant')
      momentum = momentum.reshape(g1,block_size,g2,block_size)
      momentum = momentum.transpose((0,2,1,3))

    #preconditioning momentum
    momentum = jnp.einsum('ijkl,ijln,ijnm->ijkm',precond.L,momentum,precond.R)
    g1,g2,m1,m2 = momentum.shape
    momentum = momentum.transpose((0,2,1,3)).reshape(g1*m1,g2*m2)
    momentum = momentum[:mgd_shape[0],:mgd_shape[1]]
    print("momentum shape",momentum.shape)
    momentum = momentum/jnp.linalg.norm(momentum,ord='fro')
    momentum = momentum*jnp.linalg.norm(adam_update.reshape(-1))
    return momentum.reshape(*orig_shape)

def get_paddings(s,G,block_size):
    L,R = s.L,s.R
    mgd_shape = get_merged_shape(G.shape)
    if len(mgd_shape)<=1 or sum([ dim>40000 for dim in G.shape]):
        return optax.MaskedNode()
    print(mgd_shape)
    blkd_shape = get_blocked_shape(mgd_shape,block_size)
    g1,g2,_,_ = blkd_shape
    print(mgd_shape, blkd_shape)
    padding_size_L = optax.MaskedNode()
    padding_size_R = optax.MaskedNode()
    if type(s.L).__name__!='MaskedNode':
        padding_size_L = np.ones((g1,g2),dtype=np.int32)*block_size
        if mgd_shape[0]%block_size!=0:
            padding_size_L[-1,:] = mgd_shape[0]%block_size
    if type(s.R).__name__!='MaskedNode':
        padding_size_R = np.ones((g1,g2),dtype=np.int32)*block_size
        if mgd_shape[1]%block_size!=0:
            padding_size_R[:,-1] = mgd_shape[1]%block_size

    #transpose the grid dimensions to the front
    return CasprLRPair(L=padding_size_L,R=padding_size_R)

def get_paddings_init(G,block_size):

    mgd_shape = get_merged_shape(G.shape)
    if len(mgd_shape)<=1 or sum([ dim>40000 for dim in mgd_shape]):
            return optax.MaskedNode()
    print(mgd_shape)
    blkd_shape = get_blocked_shape(mgd_shape,block_size)
    g1,g2,_,_ = blkd_shape
    print(mgd_shape, blkd_shape)

    padding_size_L = np.ones((g1,g2),dtype=np.int32)*block_size
    if mgd_shape[0]%block_size!=0:
        padding_size_L[-1,:] = mgd_shape[0]%block_size

    padding_size_R = np.ones((g1,g2),dtype=np.int32)*block_size
    if mgd_shape[1]%block_size!=0:
        padding_size_R[:,-1] = mgd_shape[1]%block_size

    #transpose the grid dimensions to the front
    return CasprLRPair(L=padding_size_L,R=padding_size_R)

def scale_by_caspr(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        matrix_epsilon: float = 1e-6,
        eps_root: float = 0.0,
        block_size: int = 1024,
        preconditioning_compute_steps: int = 20,
        start_preconditioning_step: int = 101,
        exponent_override: int = 0,
        nesterov: bool = True,
        mu_dtype: Optional[chex.ArrayDType] = None,
        caspr_p: int = -1,
        relative_epsilon: bool = True,
        inverse_type: str = 'coupled newton',
        error_tolerance: float= 1e-2,
        verbose: bool= True,
        global_grafting: bool = False,
        batch_axis_name: Any = None
        ) -> optax.GradientTransformation:
        """Rescale updates according to the Adam algorithm.


        References:
            [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)


        WARNING: PyTorch and optax's adam follow Algorithm 1 of the Kingma
            and Ba's Adam paper, if reproducing old results note that TensorFlow
            used instead the formulation just before Section 2.1 of the paper.
            See https://github.com/deepmind/optax/issues/571 for more detail.


        Args:
            b1: Decay rate for the exponentially weighted average of grads.
            b2: Decay rate for the exponentially weighted average of squared grads.
            eps: Term added to the denominator to improve numerical stability.
            eps_root: Term added to the denominator inside the square-root to improve
                numerical stability when backpropagating gradients through the rescaling.
            mu_dtype: Optional `dtype` to be used for the first order accumulator; if
                `None` then the `dtype` is inferred from `params` and `updates`.


        Returns:
            A `GradientTransformation` object.
        """




        def init_fn(params):
                mu = jax.tree_util.tree_map(  # First moment
                        lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
                nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment

                def get_padded_matrix(padding_start):
                    ix = (jnp.arange(block_size)<padding_start)
                    return (jnp.eye(block_size)*ix[jnp.newaxis,:])*ix[:,jnp.newaxis]

                get_padded_matrix_vmap = jax.vmap(get_padded_matrix,in_axes=0)

                def stat_and_precond_init(param,state_type='stats'):
                  mgd_shape = get_merged_shape(param.shape)
                  if len(param.shape) > 1 and not sum([dim>40000 for dim in param.shape]):
                      blkd_shape = get_blocked_shape(mgd_shape,block_size)
                      coeff = matrix_epsilon if state_type in ['stats','prev_stats'] else 1.0
                      jax.debug.print(state_type+' {x}',x=coeff)
                      st = jnp.zeros(blkd_shape)
                      paddings = get_paddings_init(param, block_size)

                      # jax.debug.print("padding init fn L: {L} R: {R}", L=paddings.L,R=paddings.R)
                      if mgd_shape[0]>block_size:
                        st_L = get_padded_matrix_vmap(paddings.L.reshape(-1,1))
                        st_L = st_L.reshape((blkd_shape[0],blkd_shape[1],block_size,block_size))
                      else:
                        assert blkd_shape[0]==1
                        st_L = jnp.zeros((blkd_shape[0],blkd_shape[1],mgd_shape[0],mgd_shape[0]))
                        st_L = st_L.at[:,:].set(jnp.eye(mgd_shape[0]))
                      st_L = st_L*coeff
                      if mgd_shape[1]>block_size:
                        st_R = get_padded_matrix_vmap(paddings.R.reshape(-1,1))
                        st_R = st_R.reshape((blkd_shape[0],blkd_shape[1],block_size,block_size))
                      else:
                        assert blkd_shape[1]==1
                        st_R = jnp.zeros((blkd_shape[0],blkd_shape[1],mgd_shape[1],mgd_shape[1]))
                        st_R = st_R.at[:,:].set(jnp.eye(mgd_shape[1]))
                      st_R = st_R

                      return CasprLRPair(L=st_L,R=st_R) if state_type in ['stats','preconds'] else st_L

                  else:
                      return CasprLRPair(L=optax.MaskedNode(),R=optax.MaskedNode()) if state_type in ['stats','preconds'] else optax.MaskedNode()


                stats = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='stats'), params)
                preconds = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='preconds'), params)
                prev_stats = jax.tree_util.tree_map(functools.partial(stat_and_precond_init,state_type='prev_stats'), params)

                return ScaleByCasprState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, stats=stats, preconds=preconds, prev_stats=prev_stats)


        def update_fn(updates, state, params=None):
                #TODO: start preconditioning after start_preconditioning_step
                del params
                mu = update_moment(updates, state.mu, b1, 1)
                nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
                count_inc = state.count+1
                print(state.stats)
                print(updates)
                stat_updates = jax.tree_util.tree_map(
                    lambda s,u,prevl,precondl: update_stats(s.L,s.R,prevl,precondl.L,u,block_size,b2,
                     ((jnp.maximum(count_inc-1,1))%preconditioning_compute_steps==0),matrix_epsilon),
                    state.stats,updates,state.prev_stats,state.preconds,is_leaf=lambda x: type(x).__name__=='CasprLRPair')
                # print('stat_updates ',stat_updates)
                stats = jax.tree_util.tree_map(lambda x: x.stat, stat_updates,is_leaf=lambda x: type(x).__name__=='UpdateStats')
                self_tr = lambda x: jnp.einsum("ijkk->",x)/(x.shape[0]*x.shape[1]*x.shape[2])

                prev_stats = jax.tree_util.tree_map(lambda x: x.prev_L, stat_updates,is_leaf=lambda x: type(x).__name__=='UpdateStats')
                coeffs = jax.tree_util.tree_map(lambda x: x.coeff, stat_updates,is_leaf=lambda x: type(x).__name__=='UpdateStats')
                # jax.debug.print("stats trace {x} \ncoeffs {y}",x = jax.tree_util.tree_map(self_tr,stats),y=coeffs)

                exponent = exponent_override if exponent_override !=0 else 2
                stats_paddings = jax.tree_util.tree_map(lambda s,u: get_paddings(s,u,block_size), state.stats,updates,is_leaf=lambda x: type(x).__name__=='CasprLRPair')


                preconds = jax.lax.cond((count_inc)%preconditioning_compute_steps==0, lambda: update_preconds_model(stats,state.preconds,stats_paddings,mu,
                                                                                exponent,
                                                                                matrix_epsilon,
                                                                                block_size,
                                                                                relative_epsilon,
                                                                                inverse_type,
                                                                                error_tolerance,batch_axis_name),lambda: state.preconds)
                def print_fn(m,stat_type='mu'):
                        print_fn = lambda m: jax.debug.print(
                            "step {st} " + stat_type + " l2 {x}, " + stat_type + " l0 1e-5 {y}, " + stat_type + " l0 1e-7 {z}, " + stat_type + " l0 1e-10 {u}",
                            st=count_inc, x=jnp.linalg.norm(m.reshape(-1)), y=jnp.sum(jnp.abs(m) > 1e-5),
                            z=jnp.sum(jnp.abs(m) > 1e-7), u=jnp.sum(jnp.abs(m) > 1e-10)
                            )
                if verbose:
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='mu'), mu)
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='nu'), nu)
                nu_hat = bias_correction(nu, b2, count_inc)
                def nadam_fn(m,v,g):
                        return  m / (jnp.sqrt(v + eps_root) + eps)
                def nesterov_mom(m,v,g):
                        return (b1*m+(1-b1)*g) if nesterov else m
                mu_hat = jax.tree_util.tree_map(nesterov_mom,mu,nu_hat,updates)
                mu_hat = bias_correction(mu_hat, b1, count_inc)
                adam_updates = jax.tree_util.tree_map(
                        lambda m, v, g: nadam_fn(m,v,g), mu_hat, nu_hat, updates)
                #used adam updates for rank 1 tensors, otherwise use caspr updates
                caspr_updates = jax.tree_util.tree_map(
                    lambda p,m,u: caspr_update_fn(p,m,u,block_size),
                    preconds,mu_hat,adam_updates,
                    is_leaf=lambda x: type(x).__name__=='CasprLRPair')

                updates = jax.lax.cond(count_inc>start_preconditioning_step, lambda : caspr_updates, lambda : adam_updates)
            #  jax.debug.print("preconds shape {x}",x=jax.tree_util.tree_flatten(preconds)[0][2].shape)
                if verbose:
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='updates'), updates)
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='caspr_updates'), caspr_updates)
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='adam_updates'), adam_updates)
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='stats'), stats)
                        jax.tree_util.tree_map(functools.partial(print_fn,stat_type='preconds'), preconds)
                return updates, ScaleByCasprState(count=count_inc, mu=mu, nu=nu, stats=stats, preconds=preconds, prev_stats=prev_stats)


        return optax.GradientTransformation(init_fn, update_fn)




def efficient_caspr_adaptive_full_matrix_dist_inv_optimized(
        learning_rate: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        matrix_epsilon: float = 1e-6,
        eps_root: float = 0.0,
        block_size: int = 1024,
        preconditioning_compute_steps: int = 20,
        start_preconditioning_step: int = 101,
        exponent_override: int = 0,
        nesterov: bool = True,
        mu_dtype: Optional[chex.ArrayDType] = None,
        caspr_p: int = -1,
        relative_epsilon: bool = True,
        inverse_type: str = 'eigh',
        error_tolerance: float= 1e-2,
        weight_decay: float = 1e-4,
        mask: Optional[Union[Callable[[optax.Params], Any], None]] = None,
        global_grafting: bool = False,
        batch_axis_name: Any = None
        ) -> optax.GradientTransformation:
        """Adam with weight decay regularization.


        AdamW uses weight decay to regularize learning towards small weights, as
        this leads to better generalization. In SGD you can also use L2 regularization
        to implement this as an additive loss term, however L2 regularization
        does not behave as intended for adaptive gradient algorithms such as Adam.


        References:
            Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101


        Args:
            learning_rate: A fixed global scaling factor.
            b1: Exponential decay rate to track the first moment of past gradients.
            b2: Exponential decay rate to track the second moment of past gradients.
            eps: A small constant applied to denominator outside of the square root
                (as in the Adam paper) to avoid dividing by zero when rescaling.
            eps_root: A small constant applied to denominator inside the square root (as
                in RMSProp), to avoid dividing by zero when rescaling. This is needed for
                instance when computing (meta-)gradients through Adam.
            mu_dtype: Optional `dtype` to be used for the first order accumulator; if
                `None` then the `dtype` is inferred from `params` and `updates`.
            weight_decay: Strength of the weight decay regularization. Note that this
                weight decay is multiplied with the learning rate. This is consistent
                with other frameworks such as PyTorch, but different from
                (Loshchilov et al, 2019) where the weight decay is only multiplied with
                the "schedule multiplier", but not the base learning rate.
            mask: A tree with same structure as (or a prefix of) the params PyTree,
                or a Callable that returns such a pytree given the params/updates.
                The leaves should be booleans, `True` for leaves/subtrees you want to
                apply the weight decay to, and `False` for those you want to skip. Note
                that the Adam gradient transformations are applied to all parameters.


        Returns:
            The corresponding `GradientTransformation`.
        """
        # Using jax.debug.print to print the parameters
        jax.debug.print("""
        learning_rate: {learning_rate},
        b1: {b1},
        b2: {b2},
        eps: {eps},
        matrix_epsilon: {matrix_epsilon},
        eps_root: {eps_root},
        block_size: {block_size},
        preconditioning_compute_steps: {preconditioning_compute_steps},
        start_preconditioning_step: {start_preconditioning_step},
        exponent_override: {exponent_override},
        nesterov: {nesterov},
        mu_dtype: {mu_dtype},
        caspr_p: {caspr_p},
        relative_epsilon: {relative_epsilon},
        error_tolerance: {error_tolerance},
        weight_decay: {weight_decay},
        mask: {mask},
        global_grafting: {global_grafting}
        """, learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, matrix_epsilon=matrix_epsilon,
                eps_root=eps_root, block_size=block_size, preconditioning_compute_steps=preconditioning_compute_steps,
                start_preconditioning_step=start_preconditioning_step, exponent_override=exponent_override, nesterov=nesterov,
                mu_dtype=mu_dtype, caspr_p=caspr_p, relative_epsilon=relative_epsilon,
                error_tolerance=error_tolerance, weight_decay=weight_decay, mask=mask, global_grafting=global_grafting)
        return optax.chain(
                scale_by_caspr(
                        b1, b2, eps, matrix_epsilon, eps_root, block_size,
                        preconditioning_compute_steps, start_preconditioning_step,
                        exponent_override, nesterov, mu_dtype,
                        caspr_p, relative_epsilon, inverse_type, error_tolerance,global_grafting,batch_axis_name=batch_axis_name),
                optax.add_decayed_weights(weight_decay, mask),
                _scale_by_learning_rate(learning_rate),
        )



