from typing import Union, List
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
    PRODUCT = "PRODUCT"
    BAND = "BAND"
    BOR = "BOR"
    BXOR = "BXOR"


class OpHandler:
    def __init__(self, op: Union[str, IvyReduceOp]) -> None:
        self.op = op
        if isinstance(self.op, str):
            self.op = self.op.upper()
            self.op = IvyReduceOp(self.op)

    @property
    def name(self):
        return self.op.name

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
        elif op_name == "PRODUCT":
            return t_dist.ReduceOp.PRODUCT
        elif op_name == "BAND":
            return t_dist.ReduceOp.BAND
        elif op_name == "BOR":
            return t_dist.ReduceOp.BOR
        elif op_name == "BXOR":
            return t_dist.ReduceOp.BXOR
        else:
            raise TypeError(f"Given Op: {op_name} not supported")

    @property
    def mpi_op(self):
        import mpi4py.MPI as MPI

        op_name = self.op.name
        if op_name == "SUM":
            return MPI.SUM
        elif op_name == "MEAN":
            return MPI.SUM
        elif op_name == "MAX":
            return MPI.MAX
        elif op_name == "MIN":
            return MPI.MIN
        elif op_name == "PRODUCT":
            return MPI.PROD
        elif op_name == "BAND":
            return MPI.BAND
        elif op_name == "BOR":
            return MPI.BOR
        elif op_name == "BXOR":
            return MPI.BXOR
        else:
            raise TypeError(f"Given Op: {op_name} not supported")

    def cupy_op(self):
        op_name = self.op.name
        if op_name == "SUM":
            return "sum"
        elif op_name == "MEAN":
            return "sum"
        elif op_name == "MAX":
            return "max"
        elif op_name == "MIN":
            return "min"
        elif op_name == "PRODUCT":
            return "prod"


@handle_nestable
@group_handler
@to_native_arrays_and_back
def all_reduce(
    x: Union[ivy.Array, ivy.NativeArray],
    op: Union[str, IvyReduceOp],
    *,
    group: Union[Group, None] = None,
    out: Union[ivy.Array, None] = None,
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
        x=x, op_handler=op_handler, group=group, out=out
    )


@group_handler
@handle_nestable
@to_native_arrays_and_back
def all_gather(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: int = 0,
    *,
    group: Union[Group, None] = None,
    tiled: bool = False,
    out: Union[ivy.Array, List[ivy.Array], None] = None,
) -> Union[ivy.Array, List[ivy.Array]]:
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
        x=x, axis=axis, group=group, tiled=tiled, out=out
    )


# TODO add support for input/output_split_sizes
@handle_nestable
@group_handler
@to_native_arrays_and_back
def all_to_all(
    x: Union[ivy.Array, ivy.NativeArray],
    group: Union[Group, None] = None,
    out: Union[ivy.Array, None] = None,
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
    return ivy.current_dist_backend().all_to_all(x=x, group=group, out=out)


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
    out: Union[ivy.Array, List[ivy.Array], None] = None,
) -> Union[ivy.Array, List[ivy.Array], None]:
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
        x=x, axis=axis, group=group, tiled=tiled, dst=dst, out=out
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
    out: Union[ivy.Array, None] = None,
) -> Union[ivy.Array, None]:
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
        The output of the Reduction operation on
    """
    op_handler = OpHandler(op)
    return ivy.current_dist_backend().reduce(
        x=x, op_handler=op_handler, group=group, dst=dst, out=out
    )


# TODO add support fro axis arg
@handle_nestable
@group_handler
@to_native_arrays_and_back
def scatter(
    out_buffer: Union[ivy.Array, ivy.NativeArray],
    x: Union[ivy.Array, ivy.NativeArray, None] = None,
    *,
    group: Union[Group, None] = None,
    src: int = 0,
) -> ivy.Array:
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


@handle_nestable
@group_handler
@to_native_arrays_and_back
def reduce_scatter(
    x: Union[ivy.Array, ivy.NativeArray],
    op: Union[str, IvyReduceOp],
    *,
    group: Union[Group, None] = None,
    out: Union[ivy.Array, None] = None,
) -> ivy.Array:
    """
    Preform reduce scatter operation.

    Parameters
    ----------
    x : Union[ivy.Array, ivy.NativeArray]
        _description_
    op : Union[str, IvyReduceOp]
        _description_
    group : Union[Group, None], optional
        _description_, by default None
    out : Union[ivy.Array, None], optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    op_handler = OpHandler(op)
    return ivy.current_dist_backend().reduce_scatter(
        x=x, op_handler=op_handler, group=group, out=out
    )
