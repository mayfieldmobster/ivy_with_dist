import ivy.distributed as i_dist


def new_group(ranks, **kwargs):
    return i_dist.Group(ranks)


def get_group_rank(group, global_rank):
    if isinstance(group, i_dist.Group):
        return group.ranks.index(global_rank)
    else:
        return i_dist.Group(group).ranks.index(global_rank)


def get_global_rank(group, group_rank):
    if isinstance(group, i_dist.Group):
        return group[group_rank]
    else:
        return i_dist.Group(group)[group_rank]


def get_process_group_ranks(group):
    if isinstance(group, i_dist.Group):
        return group.ranks
    else:
        return i_dist.Group(group).ranks
