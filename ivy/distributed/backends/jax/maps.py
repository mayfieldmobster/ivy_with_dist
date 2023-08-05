from typing import Union, Callable, Sequence
from functools import wraps

import jax

import ivy
import ivy.distributed as i_dist
from ivy.functional.backends.jax import JaxArray


def data_frag(*args, in_axes: Union[int, tuple], num_devices: int):
    new_args = [[] for _ in range(num_devices)]
    in_axes = (in_axes,) * len(args) if isinstance(in_axes, int) else in_axes
    if len(in_axes) != len(args):
        raise ValueError("len of in_axes must match len of args")
    for d, a in zip(in_axes, args):
        if not isinstance(a, JaxArray):
            raise TypeError("Only tensors can be mapped")
        if d is not None:
            a = ivy.split(a, num_or_size_splits=num_devices, axis=d)
            for i in range(num_devices):
                new_args[i].append(a[i])
        else:
            for i in range(num_devices):
                new_args[i].append(a)

    return new_args


def data_to_device(*args):
    return [jax.to_device(a, "gpu") for a in args]


def scalar_to_vec(*args):
    # if output is scalar convert to vecotr of shape (1,1) for all_gather
    return tuple(map(lambda x: ivy.expand_dims(x) if x.ndim() == 0 else x, args))


# TODO pmap is very similar between backends so should remove repetitive code
def pmap(
    fn: Callable,
    *,
    in_axes: Union[int, Sequence[tuple]] = 0,
    out_axes: Union[int, Sequence[tuple]] = 0,
    group: Union[i_dist.Group, None] = None,
    dst: int = 0
) -> Callable:
    comm = group

    rank = comm.Get_rank()
    num_processes = comm.Get_size()

    # not using pmap bacause mpi is easier and more consistent
    @wraps(fn)
    def _pmap(*args, **kwargs):
        if num_processes == 1:
            return ivy.vmap(fn, in_axes=in_axes, out_axes=out_axes)(*args, **kwargs)

        new_args = data_frag(*args, in_axes=in_axes, num_devices=num_processes)
        data_to_device(*new_args[rank])
        func_out = ivy.vmap(fn, in_axes=in_axes, out_axes=out_axes)(
            *new_args[rank], **kwargs
        )
        func_out = scalar_to_vec(*func_out)

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


def xmap():
    ...
