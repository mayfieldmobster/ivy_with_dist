import torch.distributed as t_dist
import ivy.distributed as i_dist


def _to_native_group(ranks):
    context = i_dist.ParallelContext()
    if context.get_world_size() == len(ranks) and context.multi_machine:
        return t_dist.group.WORLD
    return t_dist.new_group(ranks)


def _from_native_group(group):
    return t_dist.get_process_group_ranks(group)
