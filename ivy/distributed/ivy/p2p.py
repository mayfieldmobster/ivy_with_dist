from typing import Union

import ivy
from ivy.distributed.ivy.device_handeling.groups import Group
from ivy.distributed.func_wrappers import group_handler
from ivy.func_wrapper import to_native_arrays_and_back


@group_handler
@to_native_arrays_and_back
def send(
    x: Union[ivy.Array, ivy.NativeArray],
    dst: int,
    group: Union[Group, None] = None,
    tag: int = 0,
):
    """
    Send Input array from one process to another.

    Parameters
    ----------
    x : Union[ivy.Array, ivy.NativeArray]
        Input Array
    dst : int
        destineation of the input array
    group : Group
        The process group to work on. If None, the default process group will be used,
        by default None
    tag : int, optional
        the tag of the message, by default 0
    """
    ivy.current_dist_backend().send(x=x, dst=dst, tag=tag, group=group)


@group_handler
@to_native_arrays_and_back
def recv(
    x_buffer: Union[ivy.Array, ivy.NativeArray],
    src: int,
    group: Union[Group, None] = None,
    tag: int = 0,
):
    """
    Recieves data from a send operation.

    Parameters
    ----------
    x_buffer : Union[ivy.Array, ivy.NativeArray]
        A Array with the same data shape and size as the sent Array
        If there is data in the Array a empty array with the same shape
        and size will be created
    src : int
        the source of the sent array
    group : Group
        The process group to work on. If None, the default process group will be used,
        by default None
    tag : int, optional
        the tag of the message, by default 0

    Returns
    -------
    ivy.Array
        the output of the Recv operation
    """
    return ivy.current_dist_backend().recv(
        x_buffer=x_buffer, src=src, tag=tag, group=group
    )


@group_handler
def barrier(group: Union[Group, None] = None):
    """
    Preforms a barrier operation.

    Parameters
    ----------
    group : Group
        The process group to work on. If None, the default process group will be used,
        by default None
    """
    ivy.current_dist_backend().barrier(group=group)
