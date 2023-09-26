import cupy as cp
from . import _group

import ivy.distributed as i_dist
from _func_wrapper import to_dlpack_and_back, not_in_group


@to_dlpack_and_back
@not_in_group
def all_reduce(
    x: cp.ndarray,
    op_handler: i_dist.OpHandler,
    group: _group._CustomCupyGroup = None,
    out=None,
) -> cp.ndarray:
    op = op_handler.cupy_op
    if out is None:
        tensor_out = cp.empty_like(x, dtype=x.dtype)
    else:
        assert x.shape == out.shape, "given output tensor is incorrect shape"
        tensor_out = out
    group.all_reduce(x, tensor_out, op=op)
    if op_handler.op.name == "MEAN":
        tensor_out = tensor_out / cp.array(len(group), dtype=tensor_out.dtype)
    return tensor_out


@to_dlpack_and_back
@not_in_group
def all_gather(
    x: cp.ndarray,
    axis: int = 0,
    group: _group._CustomCupyGroup = None,
    tiled: bool = False,
    out=None,
):
    permutation = list(range(cp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else cp.transpose(x, permutation)
    tensor_out_shape = (len(group), *tensor_in.shape)
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
    group.all_gather(tensor_in, tensor_out, count=1)
    tensor_out = cp.concatenate(tensor_out)
    out = tensor_out if axis == 0 else cp.transpose(tensor_out, permutation)
    if tiled:
        out = cp.split(out, indices_or_sections=len(group), axis=axis)
    return out


@to_dlpack_and_back
@not_in_group
def all_to_all(
    x: cp.ndarray, group: _group._CustomCupyGroup = None, out=None
) -> cp.ndarray:
    if out is None:
        tensor_out = cp.empty_like(x, dtype=x.dtype)
    else:
        assert out.shape == x.shape, "given output tensor is incorrect shape"
        tensor_out = out
    group.all_to_all(x, tensor_out)
    return tensor_out


def broadcast(x: cp.ndarray, group: _group._CustomCupyGroup = None, src: int = 0):
    group.broadcast(x, root=src)
    return x


@to_dlpack_and_back
@not_in_group
def gather(
    x: cp.ndarray,
    axis: int = 0,
    group: _group._CustomCupyGroup = None,
    tiled: bool = False,
    dst: int = 0,
    out=None,
):
    permutation = list(range(cp.ndim(x)))
    permutation[axis] = 0
    permutation[0] = axis
    tensor_in = x if axis == 0 else cp.transpose(x, permutation)
    out_shape = (len(group), *tensor_in.shape)
    if group.g_rank == dst:
        if out is not None:
            if isinstance(out, list):
                if x.shape == out[0].shape:  # else assume already in (1,x,y,..)
                    out = list(map(lambda x: cp.expand_dims(x, axis=0)))
                tensor_out = cp.concatenate(out)
            else:
                assert out_shape == out.shape, "given output tensor is incorrect shape"
                tensor_out = out
        else:
            tensor_out = cp.empty(out_shape, dtype=x.dtype)
    else:
        tensor_out = None
    group.gather(tensor_in, tensor_out, root=dst)
    if group.g_rank == dst:
        tensor_out = cp.concatenate(tensor_out)
        out = tensor_out if axis == 0 else cp.transpose(tensor_out, permutation)
        if tiled:
            out = cp.split(out, indices_or_sections=len(group), axis=axis)
    else:
        out = tensor_out
    return out


@to_dlpack_and_back
@not_in_group
def reduce(
    x: cp.ndarray,
    op_handler: i_dist.OpHandler,
    group: _group._CustomCupyGroup = None,
    dst: int = 0,
    out=None,
):
    op = op_handler.mpi_op
    if group.g_rank == dst:
        if out is None:
            tensor_out = cp.empty_like(x, dtype=x.dtype)
        else:
            assert x.shape == out.shape, "given output tensor is incorrect shape"
            tensor_out = out
    else:
        tensor_out = None
    group.reduce(x, tensor_out, op=op, root=dst)
    if group.g_rank == dst:
        if op_handler.op.name == "MEAN" and group.g_rank == dst:
            tensor_out = tensor_out / len(group)
    return tensor_out


@to_dlpack_and_back
@not_in_group
def scatter(
    out_buffer: cp.ndarray,
    x: cp.ndarray,
    group: _group._CustomCupyGroup = None,
    src: int = 0,
):
    group.scatter(x, out_buffer, root=src)
    return out_buffer


def reduce_scatter(
    x: cp.ndarray,
    op_handler: i_dist.OpHandler,
    group: _group._CustomCupyGroup = None,
    out=None,
):
    tensor_out = []
    num_processes = len(group)
    x = cp.split(x, indices_or_sections=num_processes)
    outs = [None] * num_processes
    outs[group.g_rank] = out
    for dst, tensor_in in enumerate(x):
        tensor_out.append(
            reduce(
                tensor_in, op_handler=op_handler, group=group, dst=dst, out=outs[dst]
            )
        )

    i_dist.barrier(group=group)
    return tensor_out[group.g_rank]
