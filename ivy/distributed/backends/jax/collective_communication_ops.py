import mpi4py.MPI as MPI

import jax.numpy as jnp
import mpi4jax

import ivy
import ivy.distributed as i_dist
from ivy.functional.backends.jax import JaxArray
from ._func_wrapper import token_wrapper


def all_reduce(
    x: JaxArray, op_handler: i_dist.OpHandler, group: MPI.Comm = MPI.COMM_WORLD
) -> JaxArray:
    op = op_handler.mpi_op
    out = token_wrapper(mpi4jax.allreduce)(x, op=op, comm=group)
    if op_handler.op.name == "MEAN":
        out = out / group.Get_size()
    return out


def all_gather(
    x: JaxArray, axis: int = 0, group: MPI.Comm = MPI.COMM_WORLD, tiled: bool = False
) -> JaxArray:
    permutation = list(range(jnp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else jnp.transpose(x, axes=permutation)
    tensor_out = token_wrapper(mpi4jax.allgather)(tensor_in, comm=group)
    tensor_out = (
        tensor_out if axis == 0 else jnp.transpose(tensor_out, axes=permutation)
    )
    if tiled:
        tensor_out = ivy.split(
            tensor_out, num_or_size_splits=group.Get_size(), axis=axis
        )
    return tensor_out


def all_to_all(x: JaxArray, group: MPI.Comm = MPI.COMM_WORLD) -> JaxArray:
    return token_wrapper(mpi4jax.alltoall)(x, comm=group)


def broadcast(x: JaxArray, group: MPI.Comm = MPI.COMM_WORLD, src: int = 0):
    return token_wrapper(mpi4jax.bcast)(x=x, root=src, comm=group)


def gather(
    x: JaxArray,
    axis: int = 0,
    group: MPI.Comm = MPI.COMM_WORLD,
    tiled: bool = False,
    dst: int = 0,
):
    permutation = list(range(jnp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else jnp.transpose(x, axes=permutation)
    tensor_out = token_wrapper(mpi4jax.gather)(tensor_in, root=dst, comm=group)
    tensor_out = (
        tensor_out if axis == 0 else jnp.transpose(tensor_out, axes=permutation)
    )
    if tiled:
        tensor_out = ivy.split(
            tensor_out, num_or_size_splits=group.Get_size(), axis=axis
        )
    return tensor_out


def reduce(
    x: JaxArray,
    op_handler: i_dist.OpHandler,
    group: MPI.Comm = MPI.COMM_WORLD,
    dst: int = 0,
):
    op = op_handler.mpi_op
    tensor_out = token_wrapper(mpi4jax.reduce)(x, op=op, comm=group, root=dst)
    if op_handler.op.name == "MEAN" and group.rank == dst:
        tensor_out = tensor_out / group.Get_size()
    return tensor_out


def scatter(
    out_buffer: JaxArray, x: JaxArray, group: MPI.Comm = MPI.COMM_WORLD, src: int = 0
):
    if group.Get_rank() == src:
        out = token_wrapper(mpi4jax.scatter)(x=x, root=src, comm=group)
    else:
        out = token_wrapper(mpi4jax.scatter)(x=out_buffer, root=src, comm=group)

    return out


def reduce_scatter(
    x: JaxArray, op_handler: i_dist.OpHandler, group: MPI.Comm = MPI.COMM_WORLD
):
    x = ivy.split(x, num_or_size_splits=group.Get_size())
    for dst, tensor_in in enumerate(x):
        out = reduce(tensor_in, op_handler=op_handler, group=group, dst=dst)

    i_dist.barrier(group=group)
    return out
