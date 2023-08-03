import os


import ivy.distributed as i_dist


def init_dist():
    context = i_dist.ParallelContext()
    context.reset_context()
    context.initilize()
    os.environ["CUDA_VISIBLE_DEVICES"] = context.local_rank
    multi_machine = True if context.world_size / context.rank != 1 else False
    context.multi_machine = multi_machine
    context.default_group = i_dist.Group(range(context.world_size))
