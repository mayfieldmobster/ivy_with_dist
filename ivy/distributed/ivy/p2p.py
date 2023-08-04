from typing import Union

import ivy
from ivy.distributed.ivy.device_handeling.groups import Group
from ivy.distributed.func_wrappers import group_handler
from ivy.func_wrapper import to_native_arrays_and_back


@group_handler
@to_native_arrays_and_back
def send(x: Union[ivy.Array, ivy.NativeArray], dst: int, group: Group, tag: int = 0):
    ivy.current_dist_backend().send(x=x, dst=dst, tag=tag, group=group)


@group_handler
@to_native_arrays_and_back
def recv(
    x_buffer: Union[ivy.Array, ivy.NativeArray],
    src: int,
    group: Group,
    tag: int = 0,
):
    ivy.current_dist_backend().send(x_buffer=x_buffer, src=src, tag=tag, group=group)


@group_handler
def barrier(group: Group):
    ivy.current_dist_backend().barrier(group=group)
