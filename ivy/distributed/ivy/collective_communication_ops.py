from typing import (
    Union,
)
from enum import Enum
from functools import partial

import ivy
import ivy.distributed as i_dist
from ivy.distributed.func_wrappers import group_handler

context = i_dist.ParallelContext()


class IvyReduceOp(Enum):
    SUM = "SUM"
    MEAN = "MEAN"
    MAX = "MAX"
    MIN = "MIN"


class OpHandler:
    def __init__(self, op: Union[str, IvyReduceOp]) -> None:
        self.op = op
        if isinstance(self.op, str):
            self.op = self.op.upper()
            self.op = IvyReduceOp(self.op)

    @property
    def torch_op(self):
        import torch.distributed as t_dist

        op_name = self.op.name
        if op_name == "SUM":
            return t_dist.ReduceOp.SUM
        elif op_name == "MEAN":
            return t_dist.ReduceOp.MEAN
        elif op_name == "MAX":
            return t_dist.ReduceOp.MAX
        elif op_name == "MIN":
            return t_dist.ReduceOp.MIN

    @property
    def tensorflow_op(self):
        import tensorflow.distribute as tf_dist

        op_name = self.op.name
        if op_name == "SUM":
            return tf_dist.ReduceOp.SUM
        elif op_name == "MEAN":
            return tf_dist.ReduceOp.MEAN
        elif op_name == "MAX":
            raise ValueError("Tensorflow backend doesnt support MAX reduce op")
        elif op_name == "MIN":
            raise ValueError("Tensorflow backend doesnt support MIN reduce op")

    @property
    def jax_op(self):
        op_name = self.op.name
        if op_name == "SUM":
            return partial(ivy.sum, axis=0)
        elif op_name == "MEAN":
            return partial(ivy.mean, axis=0)
        elif op_name == "MAX":
            return partial(ivy.max, axis=0)
        elif op_name == "MIN":
            return partial(ivy.min, axis=0)


@group_handler
def all_reduce(
    x: Union[ivy.Array, ivy.NativeArray],
    op: Union[str, IvyReduceOp],
    group: Union[i_dist.Group, None] = None,
) -> ivy.Array:
    op_handler = OpHandler(op)
    return ivy.current_dist_backend().all_reduce(x, op_handler, group)


@group_handler
def all_gather(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = 0,
    group: Union[i_dist.Group, None] = None,
    tiled: bool = False,
) -> ivy.Array:
    return ivy.current_dist_backend().all_gather(x, axis, group, tiled)


@group_handler
def all_to_all(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = 0,
    group: Union[i_dist.Group, None] = None,
) -> ivy.Array:
    return ivy.current_dist_backend().all_gather(x, axis, group)


@group_handler
def gather(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = 0,
    group: Union[i_dist.Group, None] = None,
    tiled: bool = False,
    dst: int = 0,
):
    ...


@group_handler
def reduce(
    x: Union[ivy.Array, ivy.NativeArray],
    op: Union[str, IvyReduceOp],
    group: Union[i_dist.Group, None],
    dst: int = 0,
):
    ...
