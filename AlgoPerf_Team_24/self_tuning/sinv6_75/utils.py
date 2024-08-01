import jax
from flax import serialization
from typing import Any, Callable, Mapping, Optional, TypeVar, Union
import functools

T = TypeVar("T")


def load_state(path: str, state: T) -> T:
    """Load a pytree state directly from a file.

    Args:
      path: path to load pytree state from.
      state: pytree whose structure should match that of the stucture saved in the
        path. The values of this pytree are not used.

    Returns:
      The restored pytree matching the pytree structure of state.
    """
    with open(path, "rb") as fp:
        state_new = serialization.from_bytes(state, fp.read())
    tree = jax.tree_util.tree_structure(state)
    leaves_new = jax.tree_util.tree_leaves(state_new)
    return jax.tree_util.tree_unflatten(tree, leaves_new)


@functools.lru_cache(None)
def cached_jit(fn, *args, **kwargs):
    return jax.jit(fn, *args, **kwargs)
