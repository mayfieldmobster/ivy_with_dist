from typing import (
    Union,
)
from enum import Enum

import ivy
from ivy.distributed.ivy.device_handeling.groups import Group
from ivy.distributed.func_wrappers import group_handler
from ivy.func_wrapper import to_native_arrays_and_back, handle_nestable
from ivy.distributed.ivy.parallel_context import ParallelContext

context = ParallelContext()


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
            return t_dist.ReduceOp.SUM
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
        import mpi4py

        op_name = self.op.name
        if op_name == "SUM":
            return mpi4py.MPI.SUM
        elif op_name == "MEAN":
            return mpi4py.MPI.SUM
        elif op_name == "MAX":
            return mpi4py.MPI.MAX
        elif op_name == "MIN":
            return mpi4py.MPI.MIN

    @property
    def numpy_op(self):
        import mpi4py

        op_name = self.op.name
        if op_name == "SUM":
            return mpi4py.MPI.SUM
        elif op_name == "MEAN":
            return mpi4py.MPI.SUM
        elif op_name == "MAX":
            return mpi4py.MPI.MAX
        elif op_name == "MIN":
            return mpi4py.MPI.MIN


@handle_nestable
@group_handler
@to_native_arrays_and_back
def all_reduce(
    x: Union[ivy.Array, ivy.NativeArray],
    op: Union[str, IvyReduceOp],
    *,
    group: Union[Group, None] = None,
) -> ivy.Array:
    """
    Preforms an AllReduce operation.

    Parameters
    ----------
    x : Union[ivy.Array, ivy.NativeArray]
        Array Input
    op : Union[str, IvyReduceOp]
        Reduction Operator
    group : Union[Group, None], optional
        The process group to work on. If None, the default process group will be used.,
        by default None

    Returns
    -------
    ivy.Array
        Output of the Reduction Operator
    """
    op_handler = OpHandler(op)
    return ivy.current_dist_backend().all_reduce(
        x=x, op_handler=op_handler, group=group
    )


@handle_nestable
@group_handler
@to_native_arrays_and_back
def all_gather(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = 0,
    *,
    group: Union[Group, None] = None,
    tiled: bool = False,
) -> ivy.Array:
    """
    Preforms a All gather operation.

    Parameters
    ----------
    x : Union[ivy.Array, ivy.NativeArray]
        Input Array
    axis : int, optional
        Axis which the gathered tensors are concatenated across, by default 0
    group : Union[Group, None], optional
        The process group to work on. If None, the default process group will be used,
        by default None
    tiled : bool, optional
        If True the output will not be concatenated across the given axis, by default
        False

    Returns
    -------
    ivy.Array
        the output of the AllGather Operation
    """
    return ivy.current_dist_backend().all_gather(
        x=x, axis=axis, group=group, tiled=tiled
    )


# TODO add support for input/output_split_sizes
@handle_nestable
@group_handler
@to_native_arrays_and_back
def all_to_all(
    x: Union[ivy.Array, ivy.NativeArray],
    group: Union[Group, None] = None,
) -> ivy.Array:
    """
    Preforms a AlltoAll operation.

    Parameters
    ----------
    x : Union[ivy.Array, ivy.NativeArray]
        Input Array
    group : Union[Group, None], optional
        The process group to work on. If None, the default process group will be used,
        by default None

    Returns
    -------
    ivy.Array
        The output of the AlltoAll operation
    """
    return ivy.current_dist_backend().all_gather(x=x, group=group)


@handle_nestable
@group_handler
@to_native_arrays_and_back
def broadcast(
    x: Union[ivy.Array, ivy.NativeArray],
    *,
    group: Union[Group, None] = None,
    src: int = 0,
) -> ivy.Array:
    """
    Preforms a Broadcast operation.

    Parameters
    ----------
    x : Union[ivy.Array, ivy.NativeArray]
        On src rank Input Array on all other processes a buffer to recieve the data
    src : int
        source of the broadcast operation
    group : Union[Group, None]
        The process group to work on. If None, the default process group will be used,
        by default None

    Returns
    -------
    ivy.Array
        The output of the Broadcast Operation
    """
    return ivy.current_dist_backend().broadcast(x=x, group=group, src=src)


@handle_nestable
@group_handler
@to_native_arrays_and_back
def gather(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = 0,
    *,
    group: Union[Group, None] = None,
    tiled: bool = False,
    dst: int = 0,
) -> Union[ivy.Array, bool]:
    """
    Preform a Gather operation.

    Parameters
    ----------
    x : Union[ivy.Array, ivy.NativeArray]
        Input Array
    axis : int, optional
        Axis which the gathered tensors are concatenated across, by default 0
    group : Union[Group, None], optional
        The process group to work on. If None, the default process group will be used,
        by default None
    tiled : bool, optional
        If True the output will not be concatenated across the given axis, by default
        False
    dst : int, optional
        destination rank of the Gathered tensor, by default 0

    Returns
    -------
    Union[ivy.Array, bool]
        The output of the Gather operation if the processes rank == dst else True
    """
    return ivy.current_dist_backend().gather(
        x=x, axis=axis, group=group, tiled=tiled, dst=dst
    )


@handle_nestable
@group_handler
@to_native_arrays_and_back
def reduce(
    x: Union[ivy.Array, ivy.NativeArray],
    op: Union[str, IvyReduceOp],
    *,
    group: Union[Group, None] = None,
    dst: int = 0,
) -> Union[ivy.Array, bool]:
    """
    Preforms a Reuce operation.

    Parameters
    ----------
    x : Union[ivy.Array, ivy.NativeArray]
        Input Array
    op : Union[str, IvyReduceOp]
        The Reduction Operator
    group : Union[Group, None], optional
        The process group to work on. If None, the default process group will be used,
        by default None
    dst : int, optional
        destination rank of the reduced tensor, by default 0

    Returns
    -------
    Union[ivy.Array, bool]
        The ouput of the Reduction operation on
    """
    op_handler = OpHandler(op)
    return ivy.current_dist_backend().reduce(
        x=x, op_handler=op_handler, group=group, dst=dst
    )


@handle_nestable
@group_handler
@to_native_arrays_and_back
def scatter(
    out_buffer: Union[ivy.Array, ivy.NativeArray],
    x: Union[ivy.Array, ivy.NativeArray, None] = None,
    *,
    group: Union[Group, None] = None,
    src: int = 0,
):
    """
    Preforms a Scatter operation.

    Parameters
    ----------
    out_buffer : Union[ivy.Array, ivy.NativeArray]
        A array with the same shape as the excpected scattered array, must be given on
        all processes
    x : Union[ivy.Array, ivy.NativeArray, None]
        The Input Array to be scattered if rank == src else None
    group : Union[Group, None], optional
        The process group to work on. If None, the default process group will be used,
        by default None
    src : int, optional
        The root of the scatter operation, by default 0

    Returns
    -------
    _type_
        the output of the scatter operation
    """
    return ivy.current_dist_backend().scatter(
        out_buffer=out_buffer, x=x, group=group, src=src
    )


def reduce_scatter(
    x: Union[ivy.Array, ivy.NativeArray],
    op: Union[str, IvyReduceOp],
    *,
    group=Union[Group, None],
):
    OpHandler(op)
    ivy.current_dist_backend()
