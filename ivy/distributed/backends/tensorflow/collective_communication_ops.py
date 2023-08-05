import mpi4py.MPI as MPI

import cupy as cp

import ivy.distributed as i_dist
from _func_wrapper import to_dlpack_and_back


@to_dlpack_and_back
def all_reduce(
    x: cp.ndarray,
    op_handler: i_dist.OpHandler,
    group: MPI.Comm = MPI.COMM_WORLD,
    out=None,
) -> cp.ndarray:
    op = op_handler.mpi_op
    if out is None:
        tensor_out = cp.empty_like(x, dtype=x.dtype)
    else:
        assert x.shape == out.shape, "given output tensor is incorrect shape"
        tensor_out = out
    group.Allreduce(x, tensor_out, op)
    if op_handler.op.name == "MEAN":
        tensor_out = tensor_out / cp.array(group.Get_size(), dtype=tensor_out.dtype)
    return tensor_out


@to_dlpack_and_back
def all_gather(
    x: cp.ndarray,
    axis: int = 0,
    group: MPI.Comm = MPI.COMM_WORLD,
    tiled: bool = False,
    out=None,
):
    permutation = list(range(cp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else cp.transpose(x, permutation)
    tensor_out_shape = (group.Get_size(), *tensor_in.shape)
    if out is None:
        tensor_out = cp.empty(tensor_out_shape, dtype=tensor_in.dtype)
    else:
        if isinstance(out, list):
            if x.shape == out[0].shape:  # else assume already in (1,x,y,..)
                out = list(map(lambda x: cp.expand_dims(x, axis=0)))
            tensor_out = cp.concatenate(out)
        else:
            assert (
                tensor_out_shape == out.shape
            ), "given output tensor is incorrect shape"
            tensor_out = out
    group.Allgather(tensor_in, tensor_out)
    tensor_out = cp.concatenate(tensor_out)
    out = tensor_out if axis == 0 else cp.transpose(tensor_out, permutation)
    if tiled:
        out = cp.split(out, indices_or_sections=group.Get_size(), axis=axis)
    return out


@to_dlpack_and_back
def all_to_all(x: cp.ndarray, group: MPI.Comm = MPI.COMM_WORLD, out=None) -> cp.ndarray:
    if out is None:
        tensor_out = cp.empty_like(x, dtype=x.dtype)
    else:
        assert out.shape == x.shape, "given output tensor is incorrect shape"
        tensor_out = out
    group.Alltoall(x, tensor_out)
    return tensor_out


def broadcast(x: cp.ndarray, group: MPI.Comm = MPI.COMM_WORLD, src: int = 0):
    group.Bcast(x, root=src)
    return x


@to_dlpack_and_back
def gather(
    x: cp.ndarray,
    axis: int = 0,
    group: MPI.Comm = MPI.COMM_WORLD,
    tiled: bool = False,
    dst: int = 0,
    out=None,
):
    permutation = list(range(cp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else cp.transpose(x, permutation)
    out_shape = (group.Get_size(), *tensor_in.shape)
    if group.Get_rank() == dst:
        if out is None:
            if isinstance(out, list):
                if x.shape == out[0].shape:  # else assume already in (1,x,y,..)
                    out = list(map(lambda x: cp.expand_dims(x, axis=0)))
                tensor_out = cp.concatenate(out)
            else:
                assert out_shape == out.shape, "given output tensor is incorrect shape"
                tensor_out = out
    else:
        tensor_out = None
    group.Gather(tensor_in, tensor_out, root=dst)
    if group.Get_rank() == dst:
        tensor_out = cp.concatenate(tensor_out)
        out = tensor_out if axis == 0 else cp.transpose(tensor_out, permutation)
        if tiled:
            out = cp.split(out, indices_or_sections=group.Get_size(), axis=axis)
    else:
        out = tensor_out
    return out


@to_dlpack_and_back
def reduce(
    x: cp.ndarray,
    op_handler: i_dist.OpHandler,
    group: MPI.Comm = MPI.COMM_WORLD,
    dst: int = 0,
    out=None,
):
    op = op_handler.mpi_op
    if group.Get_rank() == dst:
        if out is None:
            tensor_out = cp.empty_like(x, dtype=x.dtype)
        else:
            assert x.shape == out.shape, "given output tensor is incorrect shape"
            tensor_out = out
    else:
        tensor_out = None
    group.Reduce(x, tensor_out, op=op, root=dst)
    if group.Get_rank() == dst:
        if op_handler.op.name == "MEAN" and group.rank == dst:
            tensor_out = tensor_out / group.Get_size()
    return tensor_out


@to_dlpack_and_back
def scatter(
    out_buffer: cp.ndarray,
    x: cp.ndarray,
    group: MPI.Comm = MPI.COMM_WORLD,
    src: int = 0,
):
    group.Scatter(x, out_buffer, root=src)
    return out_buffer


def reduce_scatter(
    x: cp.ndarray,
    op_handler: i_dist.OpHandler,
    group: MPI.Comm = MPI.COMM_WORLD,
    out=None,
):
    tensor_out = []
    x = cp.split(x, indices_or_sections=group.Get_size())
    for dst, tensor_in in enumerate(x):
        tensor_out.append(
            reduce(tensor_in, op_handler=op_handler, group=group, dst=dst, out=out)
        )

    i_dist.barrier(group=group)
    return tensor_out[group.Get_rank()]
