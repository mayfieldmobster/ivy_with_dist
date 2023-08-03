import mpi4py.MPI as MPI

import numpy as np

import ivy
import ivy.distributed as i_dist


def all_reduce(
    x: np.ndarray, op_handler: i_dist.OpHandler, group: MPI.Comm = MPI.COMM_WORLD
) -> np.ndarray:
    op = op_handler.numpy_op
    tensor_out = np.empty_like(x, dtype=x.dtype)
    group.Allreduce(x, tensor_out, op)
    if op_handler.op.name == "MEAN":
        tensor_out = tensor_out / np.array(group.Get_size(), dtype=tensor_out.dtype)
    return tensor_out


def all_gather(
    x: np.ndarray, axis: int = 0, group: MPI.Comm = MPI.COMM_WORLD, tiled: bool = False
):
    permutation = list(range(np.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else x.transpose(permutation)
    tensor_out_shape = (group.Get_size(), *x.shape)
    tensor_out = np.empty(tensor_out_shape, dtype=tensor_in.dtype)
    group.Allgather(tensor_in, tensor_out)
    tensor_out = ivy.concat(tensor_out)
    out = tensor_out if axis == 0 else tensor_out.transpose(permutation)
    if tiled:
        out = ivy.split(out, num_or_size_splits=group.Get_size(), axis=axis)
    return out


def all_to_all(x: np.ndarray, group: MPI.Comm = MPI.COMM_WORLD) -> np.ndarray:
    tensor_out = np.empty_like(x, dtype=x.dtype)
    group.Alltoall(x, tensor_out)
    return tensor_out


def gather(
    x: np.ndarray,
    axis: int = 0,
    group: MPI.Comm = MPI.COMM_WORLD,
    tiled: bool = False,
    dst: int = 0,
):
    permutation = list(range(np.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else x.transpose(permutation)
    out_shape = (group.Get_size(), *x.shape)
    tensor_out = np.empty(out_shape, dtype=x.dtype)
    group.Gather(tensor_in, tensor_out, root=dst)
    tensor_out = ivy.concat(tensor_out)
    out = tensor_out if axis == 0 else tensor_out.transpose(permutation)
    if tiled:
        out = ivy.split(out, num_or_size_splits=group.Get_size(), axis=axis)
    return out


def reduce(
    x: np.ndarray,
    op_handler: i_dist.OpHandler,
    group: MPI.Comm = MPI.COMM_WORLD,
    dst: int = 0,
):
    op = op_handler.numpy_op
    tensor_out = np.empty_like(x, dtype=x.dtype)
    group.Reduce(x, tensor_out, op=op, root=dst)
    if op_handler.op.name == "MEAN" and group.rank == dst:
        tensor_out = tensor_out / group.Get_size()
    return tensor_out
