from typing import Union

import ivy
import ivy.distributed as i_dist


def send(
    x: Union[ivy.Array, ivy.NativeArray], dst: int, group: i_dist.Group, tag: int = 0
):
    ivy.current_dist_backend().send(x=x, dst=dst, tag=tag, group=group)


def recv(
    x_buffer: Union[ivy.Array, ivy.NativeArray],
    src: int,
    group: i_dist.Group,
    tag: int = 0,
):
    ivy.current_dist_backend().send(x_buffer=x_buffer, src=src, tag=tag, group=group)


def sendrecv():
    ...
