from typing import (
    Union,
    Callable,
    Sequence,
)

import ivy
from ivy.distributed.ivy.device_handeling.groups import Group
from ivy.distributed.func_wrappers import group_handler


@group_handler
def pmap(
    fn: Callable,
    # axis_name: Hashable,  #TODO add support in the fututre
    *,
    in_axes: Union[int, Sequence[tuple]] = 0,
    out_axes: Union[int, Sequence[tuple]] = 0,
    group: Union[Group, None] = None,
    dst: int = 0
) -> Callable:
    return ivy.current_dist_backend().pmap(
        fn, in_axes=in_axes, out_axes=out_axes, group=group, dst=dst
    )


def xmap():
    ...
