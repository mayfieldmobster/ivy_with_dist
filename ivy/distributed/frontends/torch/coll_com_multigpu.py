# multi GPU is depreciated
# you dont need to call a multigpu function to run on multiple GPUs


from ivy.utils.exceptions import IvyNotImplementedException

from . import coll_com


def broadcast_multigpu(tensor_list, src, group=None, async_op=False, src_tensor=0):
    raise IvyNotImplementedException


def all_reduce_multigpu(
    tensor_list, op=coll_com.ReduceOp.SUM, group=None, async_op=False
):
    raise IvyNotImplementedException
