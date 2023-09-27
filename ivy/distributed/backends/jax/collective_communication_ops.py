import mpi4py.MPI as MPI

import jax.numpy as jnp
import mpi4jax

import ivy
import ivy.distributed as i_dist
from ivy.functional.backends.jax import JaxArray
from ._func_wrapper import token_wrapper


def all_reduce(
    x: JaxArray,
    op_handler: i_dist.OpHandler,
    group: MPI.Comm = MPI.COMM_WORLD,
    out=None,
) -> JaxArray:
    op = op_handler.mpi_op
    if out is None:
        out = ivy.empty_like(x, dtype=x.dtype)
    out.inplace_update(token_wrapper(mpi4jax.allreduce)(x, op=op, comm=group))
    # out[:] = token_wrapper(mpi4jax.allreduce)(x, op=op, comm=group)
    if op_handler.op.name == "MEAN":
        out = out / group.Get_size()
    return out


def all_gather(
    x: JaxArray,
    axis: int = 0,
    group: MPI.Comm = MPI.COMM_WORLD,
    tiled: bool = False,
    out=None,
) -> JaxArray:
    permutation = list(range(jnp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else jnp.transpose(x, axes=permutation)
    tensor_out_shape = (group.Get_size(), *tensor_in.shape)
    if out is None:
        tensor_out = jnp.empty(tensor_out_shape, dtype=tensor_in.dtype)
    else:
        if isinstance(out, list):
            if x.shape == out[0].shape:  # else assume already in (1,x,y,..)
                out = list(map(lambda x: ivy.expand_dims(x, axis=0)))
            tensor_out = ivy.concat(out)
        else:
            assert (
                tensor_out_shape == out.shape
            ), "given output tensor is incorrect shape"
            tensor_out = out
    tensor_out = token_wrapper(mpi4jax.allgather)(tensor_in, comm=group)
    tensor_out = ivy.concat(tensor_out)
    out = tensor_out if axis == 0 else jnp.transpose(x, permutation)
    if tiled:
        out = ivy.split(out, num_or_size_splits=group.Get_size(), axis=axis)
    return out


def all_to_all(x: JaxArray, group: MPI.Comm = MPI.COMM_WORLD, out=None) -> JaxArray:
    if out is None:
        out = ivy.empty_like(x, dtype=x.dtype)
    out.inplace_update(token_wrapper(mpi4jax.alltoall)(x, comm=group))
    # out[:] = token_wrapper(mpi4jax.alltoall)(x, comm=group)
    return out


def broadcast(x: JaxArray, group: MPI.Comm = MPI.COMM_WORLD, src: int = 0):
    return token_wrapper(mpi4jax.bcast)(x=x, root=src, comm=group)


def gather(
    x: JaxArray,
    axis: int = 0,
    group: MPI.Comm = MPI.COMM_WORLD,
    tiled: bool = False,
    dst: int = 0,
    out=None,
):
    permutation = list(range(jnp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else jnp.transpose(x, permutation)
    out_shape = (group.Get_size(), *tensor_in.shape)
    if group.Get_rank() == dst:
        if out is None:
            out = ivy.empty(out_shape, dtype=x.dtype)
        else:
            if isinstance(out, list):
                if x.shape == out[0].shape:  # else assume already in (1,x,y,..)
                    out = list(map(lambda x: ivy.expand_dims(x, axis=0)))
                out = ivy.concat(out)
            else:
                assert out_shape == out.shape, "given output tensor is incorrect shape"
            out = ivy.Array(out)

    else:
        out = None
    tensor_out = token_wrapper(mpi4jax.gather)(tensor_in, root=dst, comm=group)
    if group.Get_rank() == dst:
        out.inplace_update(tensor_out)
        out = ivy.concat(out)
        out = out if axis == 0 else jnp.transpose(out, permutation)
        if tiled:
            out = ivy.split(out, num_or_size_splits=group.Get_size(), axis=axis)
    return out

    if group.Get_rank() == dst:
        if out is None:
            out = ivy.empty((group.Get_size(), *x.shape), dtype=x.dtype)
    permutation = list(range(jnp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else jnp.transpose(x, axes=permutation)
    tensor_out = token_wrapper(mpi4jax.gather)(tensor_in, root=dst, comm=group)
    if group.Get_rank() == dst:
        out.inplace_update(
            tensor_out if axis == 0 else jnp.transpose(tensor_out, axes=permutation)
        )
        # out[:] = (
        #    tensor_out if axis == 0 else jnp.transpose(tensor_out, axes=permutation)
        # )
        if tiled:
            out = ivy.split(out, num_or_size_splits=group.Get_size(), axis=axis)
    return out


def reduce(
    x: JaxArray,
    op_handler: i_dist.OpHandler,
    group: MPI.Comm = MPI.COMM_WORLD,
    dst: int = 0,
    out=None,
):
    if group.Get_rank() == dst:
        if out is None:
            out = ivy.empty_like(x, dtype=x.dtype)
    op = op_handler.mpi_op
    tensor_out = token_wrapper(mpi4jax.reduce)(x, op=op, comm=group, root=dst)
    if group.Get_rank() == dst:
        out.inplace_update(tensor_out)
        # out[:] = tensor_out
        if op_handler.op.name == "MEAN" and group.rank == dst:
            out = out / group.Get_size()
    return out


def scatter(
    out_buffer: JaxArray, x: JaxArray, group: MPI.Comm = MPI.COMM_WORLD, src: int = 0
):
    if group.Get_rank() == src:
        out = token_wrapper(mpi4jax.scatter)(x=x, root=src, comm=group).reshape(x.shape)
    else:
        out = token_wrapper(mpi4jax.scatter)(
            x=out_buffer, root=src, comm=group
        ).reshape(out_buffer.shape)
    return out


def reduce_scatter(
    x: JaxArray,
    op_handler: i_dist.OpHandler,
    group: MPI.Comm = MPI.COMM_WORLD,
    out=None,
):
    tensor_out = []
    num_processes = group.Get_size()
    x = ivy.split(x, num_or_size_splits=num_processes)
    outs = [None] * num_processes
    outs[group.Get_rank()] = out
    for dst, tensor_in in enumerate(x):
        tensor_out.append(
            reduce(
                tensor_in.data,
                op_handler=op_handler,
                group=group,
                dst=dst,
                out=outs[dst],
            )
        )
    i_dist.barrier(group=group)
    return tensor_out[group.Get_rank()]
