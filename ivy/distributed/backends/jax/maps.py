from typing import Union, Callable, Hashable, Sequence

import ivy.distributed as i_dist

import jax


def pmap(
    fn: Callable,
    axis_name: Hashable,
    *,
    in_axes: Union[int, Sequence[tuple]] = 0,
    out_axes: Union[int, Sequence[tuple]] = 0,
    group: Union[i_dist.Group, None] = None,
    dst: int = 0
) -> Callable:
    devices = group.ranks_to_jax_devices()

    def wrapper(*args, **kwargs):
        fn_out = jax.pmap(
            fn, axis_name=axis_name, in_axes=in_axes, out_axes=out_axes, devices=devices
        )(*args, **kwargs)
        if dst == -1:
            out = jax.device_put_replicated(fn_out, devices=devices)
            return out
        else:
            out = jax.device_put(fn_out, dst)
            return out

    return wrapper


def xmap():
    ...
