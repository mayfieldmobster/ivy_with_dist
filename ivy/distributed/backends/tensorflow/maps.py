from typing import Union, Sequence, Callable
import mpi4py.MPI as MPI
from functools import wraps

import cupy as cp

import ivy
import ivy.distributed as i_dist
from ivy.func_wrapper import to_native_arrays_and_back
from _func_wrapper import to_dlpack_and_back


def data_frag(*args, in_axes: Union[int, Sequence[int]], num_devices: int):
    new_args = [[] for _ in range(num_devices)]
    in_axes = (in_axes,) * len(args) if isinstance(in_axes, int) else in_axes
    if len(in_axes) != len(args):
        raise ValueError("len of in_axes must match len of args")
    for d, a in zip(in_axes, args):
        if not isinstance(a, cp.ndarray):
            raise TypeError("Only tensors can be mapped")
        if d is not None:
            a = ivy.split(a, num_or_size_splits=num_devices, axis=d)
            for i in range(num_devices):
                new_args[i].append(a[i])
        else:
            for i in range(num_devices):
                new_args[i].append(a)

    return new_args


# could use mpi map as a replacement may even be faster
def pmap(
    fn: Callable,
    *,
    in_axes: Union[int, Sequence[tuple]] = 0,
    out_axes: Union[int, Sequence[tuple]] = 0,
    group: MPI.Comm = MPI.COMM_WORLD,
    dst: int = 0
) -> Callable:
    comm = group

    rank = comm.Get_rank()
    num_processes = comm.Get_size()

    # TODO this wont work as ivy.vamp doesnt support cupy
    @wraps(fn)
    @to_dlpack_and_back
    @to_native_arrays_and_back
    def _pmap(*args, **kwargs):
        if num_processes == 1:
            return ivy.vmap(fn, in_axes=in_axes, out_axes=out_axes)(*args, **kwargs)

        new_args = data_frag(*args, in_axes=in_axes, num_devices=num_processes)
        func_out = ivy.vmap(fn, in_axes=in_axes, out_axes=out_axes)(
            *new_args[rank], **kwargs
        )

        out_axes_ = (
            (out_axes,) * len(func_out) if isinstance(out_axes, int) else out_axes
        )

        if dst == -1:
            if isinstance(func_out, tuple):
                out = []
                for i in range(len(func_out)):
                    out.append(
                        i_dist.all_gather(func_out[i], out_axes_[i], group=group)
                    )
                return tuple(out)
            else:
                return i_dist.all_gather(func_out, out_axes_[0], group=group)
        else:
            if isinstance(func_out, tuple):
                out = []
                for i in range(len(func_out)):
                    out.append(
                        i_dist.gather(
                            func_out[i], out_axes_[i], group=group, tiled=True, dst=dst
                        )
                    )
                    return tuple(out)
            else:
                return i_dist.gather(func_out, out_axes_[0], group=group, dst=dst)

    return _pmap
