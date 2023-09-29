import torch.distributed as t_dist
import ivy.distributed as i_dist


def _to_native_group(ranks):
    context = i_dist.ParallelContext()
    if list(range(context.get_world_size())) == ranks:
        return t_dist.group.WORLD
    return t_dist.new_group(ranks)


def _from_native_group(group: t_dist.ProcessGroup):
    return t_dist.get_process_group_ranks(group)


def _rank(group: t_dist.ProcessGroup):
    return group.rank()