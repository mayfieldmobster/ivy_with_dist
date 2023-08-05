import mpi4py.MPI as MPI

import numpy as np

import ivy
import ivy.distributed as i_dist


def all_reduce(
    x: np.ndarray,
    op_handler: i_dist.OpHandler,
    group: MPI.Comm = MPI.COMM_WORLD,
    out=None,
) -> np.ndarray:
    op = op_handler.mpi_op
    if out is None:
        tensor_out = np.empty_like(x, dtype=x.dtype)
    else:
        assert x.shape == out.shape, "given output tensor is incorrect shape"
        tensor_out = out
    group.Allreduce(x, tensor_out, op)
    if op_handler.op.name == "MEAN":
        tensor_out = tensor_out / np.array(group.Get_size(), dtype=tensor_out.dtype)
    return tensor_out


def all_gather(
    x: np.ndarray,
    axis: int = 0,
    group: MPI.Comm = MPI.COMM_WORLD,
    tiled: bool = False,
    out=None,
):
    permutation = list(range(ivy.get_num_dims(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else np.transpose(x, permutation)
    tensor_out_shape = (group.Get_size(), *tensor_in.shape)
    if out is None:
        tensor_out = np.empty(tensor_out_shape, dtype=tensor_in.dtype)
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
    group.Allgather(tensor_in, tensor_out)
    out = tensor_out if axis == 0 else np.transpose(x, permutation)
    if tiled:
        out = ivy.split(out, num_or_size_splits=group.Get_size(), axis=axis)
    return out


def all_to_all(x: np.ndarray, group: MPI.Comm = MPI.COMM_WORLD, out=None) -> np.ndarray:
    if out is None:
        tensor_out = np.empty_like(x, dtype=x.dtype)
    else:
        assert out.shape == x.shape, "given output tensor is incorrect shape"
        tensor_out = out
    group.Alltoall(x, tensor_out)
    return tensor_out


def broadcast(x: np.ndarray, group: MPI.Comm = MPI.COMM_WORLD, src: int = 0):
    group.Bcast(x, root=src)
    return x


def gather(
    x: np.ndarray,
    axis: int = 0,
    group: MPI.Comm = MPI.COMM_WORLD,
    tiled: bool = False,
    dst: int = 0,
    out=None,
):
    permutation = list(range(np.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else np.transpose(x, permutation)
    out_shape = (group.Get_size(), *tensor_in.shape)
    if group.Get_rank() == dst:
        if out is None:
            tensor_out = np.empty(out_shape, dtype=x.dtype)
        else:
            if isinstance(out, list):
                if x.shape == out[0].shape:  # else assume already in (1,x,y,..)
                    out = list(map(lambda x: ivy.expand_dims(x, axis=0)))
                tensor_out = ivy.concat(out)
            else:
                assert out_shape == out.shape, "given output tensor is incorrect shape"
                tensor_out = out
    else:
        tensor_out = None
    group.Gather(tensor_in, tensor_out, root=dst)
    if group.Get_rank() == dst:
        tensor_out = ivy.concat(tensor_out)
        out = tensor_out if axis == 0 else np.transpose(tensor_out, permutation)
        if tiled:
            out = ivy.split(out, num_or_size_splits=group.Get_size(), axis=axis)
    else:
        out = tensor_out
    return out


def reduce(
    x: np.ndarray,
    op_handler: i_dist.OpHandler,
    group: MPI.Comm = MPI.COMM_WORLD,
    dst: int = 0,
    out=None,
):
    op = op_handler.mpi_op
    if group.Get_rank() == dst:
        if out is None:
            tensor_out = np.empty_like(x, dtype=x.dtype)
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


def scatter(
    out_buffer: np.ndarray,
    x: np.ndarray,
    group: MPI.Comm = MPI.COMM_WORLD,
    src: int = 0,
):
    group.Scatter(x, out_buffer, root=src)
    return out_buffer


def reduce_scatter(
    x: np.ndarray,
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
                tensor_in, op_handler=op_handler, group=group, dst=dst, out=outs[dst]
            )
        )

    i_dist.barrier(group=group)
    return tensor_out[group.Get_rank()]
