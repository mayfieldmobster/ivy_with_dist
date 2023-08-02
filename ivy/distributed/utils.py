from ivy.distributed.ivy.parallel_context.parallel_context import ParallelContext


def is_initialized():
    return ParallelContext().is_initized
