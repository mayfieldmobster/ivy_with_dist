from typing import Union, Optional

import ivy.distributed as i_dist

from ivy.utils.exceptions import IvyNotImplementedException

context = i_dist.ParallelContext()


def get_backend(group: Optional[Union[i_dist.Group, i_dist.NativeGroup, None]] = None):
    raise IvyNotImplementedException


def get_rank(group: Optional[Union[i_dist.Group, i_dist.NativeGroup, None]] = None):
    if group is None:
        return context.get_global_rank()
    elif isinstance(group, i_dist.Group):
        return group.rank
    elif isinstance(group, i_dist.NativeGroup):
        return i_dist.Group(group).rank
    else:
        TypeError(
            "Group must be of type: Union[i_dist.Group, i_dist.NativeGroup, None]"
        )


def get_world_size(
    group: Optional[Union[i_dist.Group, i_dist.NativeGroup, None]] = None
):
    if group is None:
        return context.get_world_size()
    elif isinstance(group, i_dist.Group):
        return group.size
    elif isinstance(group, i_dist.NativeGroup):
        return i_dist.Group(group).size
    else:
        TypeError(
            "Group must be of type: Union[i_dist.Group, i_dist.NativeGroup, None]"
        )
