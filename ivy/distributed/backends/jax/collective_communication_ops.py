import jax.numpy as jnp
import mpi4jax

import ivy
import ivy.distributed as i_dist
from ivy.functional.backends.jax import JaxArray
from ._func_wrapper import token_wrapper

context = i_dist.ParallelContext()


def all_reduce(
    x: JaxArray, op_handler: i_dist.OpHandler, group: i_dist.Group = None
) -> JaxArray:
    comm = group.ranks_to_jax_devices
    op = op_handler.jax_op
    return token_wrapper(mpi4jax.allreduce)(x, op=op, comm=comm)


def all_gather(
    x: JaxArray, axis: int = 0, group: i_dist.Group = None, tiled: bool = False
) -> JaxArray:
    comm = group.ranks_to_jax_devices
    permutation = list(range(jnp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else jnp.transpose(x, axes=permutation)
    out = token_wrapper(mpi4jax.allgather)(tensor_in, comm=comm)
    out = out if axis == 0 else jnp.transpose(out, axes=permutation)
    if tiled:
        out = ivy.split(out, num_or_size_splits=comm.Get_size(), axis=axis)
    return out


def all_to_all(x: JaxArray, axis: int = 0, group: i_dist.Group = None) -> JaxArray:
    ...


def gather(
    x: JaxArray,
    axis: int = 0,
    group: i_dist.Group = None,
    tiled: bool = False,
    dst: int = 0,
):
    comm = group.ranks_to_jax_devices
    permutation = list(range(jnp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else jnp.transpose(x, axes=permutation)
    out = token_wrapper(mpi4jax.gather)(tensor_in, root=dst, comm=comm)
    out = out if axis == 0 else jnp.transpose(out, axes=permutation)
    if tiled:
        out = ivy.split(out, num_or_size_splits=comm.Get_size(), axis=axis)
    return out


def reduce(
    x: JaxArray,
    op: i_dist.OpHandler,
    group: i_dist.Group = None,
    dst: int = 0,
):
    ...
