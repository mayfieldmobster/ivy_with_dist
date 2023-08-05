from enum import Enum

import ivy
import ivy.distributed as i_dist

from ivy.utils.exceptions import IvyNotImplementedException


class ReduceOp(Enum):
    SUM = 0
    PRODUCT = -1
    MAX = 2
    MIN = 3
    BAND = -4
    BOR = -5
    BXOR = -6
    PREMUL_SUM = -7
    AVG = 8


def _op_to_ophandler(op: ReduceOp):
    if op.value < 0:
        raise Exception(f"Reduction Op {op.name} is not supported yet")
    op = op.name if op.name != "AVG" else "MEAN"
    return i_dist.OpHandler(op)


def broadcast(tensor, src, group=None, async_op=False):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")
    return i_dist.broadcast(tensor, src=src, group=group)


def broadcast_object_list(object_list, src=0, group=None, device=None):
    raise IvyNotImplementedException


def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")
    op = _op_to_ophandler(op)
    return i_dist.all_reduce(x=tensor, op=op, group=group)


def reduce(tensor, dst, op=ReduceOp.SUM, group=None, async_op=False):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")
    op = _op_to_ophandler(op)
    return i_dist.reduce(x=tensor, op=op, group=group, dst=dst)


def all_gather(tensor_list, tensor, group=None, async_op=False):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")
    return i_dist.all_gather(x=tensor, axis=0, group=group, tiled=False)


def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")
    return i_dist.all_gather(x=input_tensor, axis=0, group=group, tiled=False)


def all_gather_object(object_list, obj, group=None):
    raise IvyNotImplementedException


def gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")
    return i_dist.gather(x=tensor, axis=0, group=group, tiled=False, dst=dst)


def gather_object(obj, object_gather_list=None, dst=0, group=None):
    raise IvyNotImplementedException


def scatter(tensor, scatter_list=None, src=0, group=None, async_op=False):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")
    return i_dist.scatter(out_buffer=tensor, x=scatter_list, group=group, src=src)


def scatter_object_list(
    scatter_object_output_list, scatter_object_input_list, src=0, group=None
):
    raise IvyNotImplementedException


def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False):
    # Naive implementation, TODO implement efficient version
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")
    op = _op_to_ophandler(op)
    out = i_dist.reduce(x=input_list, op=op, group=group, dst=0)
    return i_dist.scatter(out_buffer=output, x=out, group=group, src=0)


def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")
    op = _op_to_ophandler(op)
    out = i_dist.reduce(x=input, op=op, group=group, dst=0)
    return i_dist.scatter(out_buffer=output, x=out, group=group, src=0)


def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")

    if output_split_sizes or input_split_sizes:
        raise Exception(
            "output_split_sizes and input_split_sizes args not supported yet by ivy"
        )

    return i_dist.all_to_all(input, group=group)


def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")

    return i_dist.all_to_all(x=input_tensor_list, group=group)


def barrier(group=None, async_op=False, device_ids=None):
    if async_op:
        ivy.warn("Asynchronous Implementations not supported yet")
    i_dist.barrier(group=group)


def monitored_barrier(group=None, timeout=None, wait_all_ranks=False):
    raise IvyNotImplementedException
