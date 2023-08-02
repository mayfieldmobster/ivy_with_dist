from typing import Union

import ivy
from ivy.distributed.ivy.device_handeling.groups import Group
from ivy.distributed.func_wrappers import group_none_handler, group_to_native


@group_to_native
@group_none_handler
def send(x: Union[ivy.Array, ivy.NativeArray], dst: int, group: Group, tag: int = 0):
    ivy.current_dist_backend().send(x=x, dst=dst, tag=tag, group=group)


@group_to_native
@group_none_handler
def recv(
    x_buffer: Union[ivy.Array, ivy.NativeArray],
    src: int,
    group: Group,
    tag: int = 0,
):
    ivy.current_dist_backend().send(x_buffer=x_buffer, src=src, tag=tag, group=group)


def sendrecv():
    ...
