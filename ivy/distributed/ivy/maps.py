from typing import (
    Union,
    Callable,
    Hashable,
    Sequence,
)

import ivy
import ivy.distributed as i_dist


def pmap(
    fn: Callable,
    axis_name: Hashable,
    *,
    in_axes: Union[int, Sequence[tuple]] = 0,
    out_axes: Union[int, Sequence[tuple]] = 0,
    group: Union[i_dist.Group, None] = None,
    dst: int = 0
) -> Callable:
    return ivy.current_dist_backend().pmap(
        fn, axis_name, in_axes=in_axes, out_axes=out_axes, group=group, dst=dst
    )


def xmap():
    ...