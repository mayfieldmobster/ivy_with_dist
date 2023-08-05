from ivy.distributed.ivy.parallel_context.parallel_context import ParallelContext


def is_initialized():
    return ParallelContext().is_initized


def print0(*args, group=None, **kwargs):
    if group is None:
        rank = ParallelContext().rank
    else:
        rank = group.rank
    if rank == 0:
        print(*args, **kwargs)
