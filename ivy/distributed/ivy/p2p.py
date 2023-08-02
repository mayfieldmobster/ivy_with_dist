from typing import Union

import ivy
from ivy.distributed.ivy.device_handeling.groups import Group


def send(x: Union[ivy.Array, ivy.NativeArray], dst: int, group: Group, tag: int = 0):
    ivy.current_dist_backend().send(x=x, dst=dst, tag=tag, group=group)


def recv(
    x_buffer: Union[ivy.Array, ivy.NativeArray],
    src: int,
    group: Group,
    tag: int = 0,
):
    ivy.current_dist_backend().send(x_buffer=x_buffer, src=src, tag=tag, group=group)


def sendrecv():
    ...
