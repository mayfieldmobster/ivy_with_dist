import mpi4py.MPI as MPI

import numpy as np

import ivy
import ivy.distributed as i_dist


def all_reduce(
    x: np.ndarray, op_handler: i_dist.OpHandler, group: MPI.Comm = MPI.COMM_WORLD
) -> np.ndarray:
    op = op_handler.numpy_op
    out = np.empty_like(x)
    group.Allreduce(x, out, op)
    if op_handler.op.name == "MEAN":
        out = out / np.array(group.Get_size(), dtype=out.dtype)
    return out


def all_gather(
    x: np.ndarray, axis: int = 0, group: MPI.Comm = MPI.COMM_WORLD, tiled: bool = False
):
    permutation = list(range(np.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else x.transpose(permutation)
    tensor_out_shape = (group.Get_size(), *x.shape)
    tensor_out = np.empty(tensor_out_shape)
    group.Allgather(tensor_in, tensor_out)
    tensor_out = ivy.concat(tensor_out)
    out = tensor_out if axis == 0 else tensor_out.transpose(permutation)
    if tiled:
        out = ivy.split(out, num_or_size_splits=group.Get_size(), axis=axis)
    return out


def all_to_all(
    x: np.ndarray, axis: int = 0, group: MPI.Comm = MPI.COMM_WORLD
) -> np.ndarray:
    ...


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
    tensor_out = np.empty(out_shape)
    group.Gather(tensor_in, tensor_out, root=dst)
    tensor_out = ivy.concat(tensor_out)
    out = tensor_out if axis == 0 else tensor_out.transpose(permutation)
    if tiled:
        out = ivy.split(out, num_or_size_splits=group.Get_size(), axis=axis)
    return out
