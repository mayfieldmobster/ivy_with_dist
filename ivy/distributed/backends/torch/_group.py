import torch.distributed as t_dist
import ivy.distributed as i_dist


def _to_native_group(ranks):
    context = i_dist.ParallelContext()
    if context.get_world_size() == len(ranks):
        return t_dist.group.WORLD
    return t_dist.new_group(ranks)
