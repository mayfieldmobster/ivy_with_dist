import ivy.distributed as i_dist


def init_dist(**kwargs):
    context = i_dist.ParallelContext()
    context.reset_context()
    context.initilize()
    context.backend = "mpi"
    context.default_group = i_dist.Group(range(context.world_size))
