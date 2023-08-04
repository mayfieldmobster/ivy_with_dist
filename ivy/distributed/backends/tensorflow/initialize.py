import os


import ivy.distributed as i_dist


def init_dist():
    context = i_dist.ParallelContext()
    context.reset_context()
    context.initilize()
    os.environ["CUDA_VISIBLE_DEVICES"] = context.local_rank
    context.default_group = i_dist.Group(range(context.world_size))
