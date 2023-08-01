from typing import Sequence

import ivy.distributed as i_dist


def _prod(*args):
    p = 1
    for arg in args:
        p *= arg
    return p


class Mesh:
    def __init__(self, shape: Sequence[int], group: i_dist.Group):
        if len(group.ranks) != _prod(*shape):
            # TODO update error type
            raise Exception("shape should be able to fit all devices in group")
        self.shape = shape
        self.group = group


class NamedMesh(Mesh):
    def __init__(
        self, shape: Sequence[int], group: i_dist.Group, axis_names: Sequence[str]
    ):
        if len(shape) != len(axis_names):
            raise Exception("number of axis must equal number of names")
        super().__init__(shape=shape, group=group)
        self.axis_names = axis_names


class MpDpMesh(NamedMesh):
    def __init__(self, shape: Sequence[int], group: i_dist.Group):
        if len(shape) != 2:
            raise Exception("shape must only have 2 axis eg (4,2)")
        axis_names = ("dp", "mp")
        super().__init__(shape, group, axis_names)
