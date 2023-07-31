from typing import Union, Callable, Hashable, Sequence
from functools import wraps

import tensorflow as tf

import ivy.distributed as i_dist
import ivy

context = i_dist.ParallelContext()


def data_frag(*args, in_axes: Union[int, tuple], num_devices: int):
    new_args = [[] for _ in range(num_devices)]
    if len(in_axes) != len(args):
        raise ValueError("len of in_axes must match len of args")
    for d, a in zip(in_axes, args):
        if not isinstance(a, tf.Tensor) and not isinstance(a, tf.Variable):
            raise TypeError("Only tensors can be mapped")
        if d is not None:
            a = ivy.split(a, num_or_size_splits=num_devices, axis=d)
            for i in range(num_devices):
                new_args[i].append(a[i])
        else:
            for i in range(num_devices):
                new_args[i].append(a)

    return new_args


def data_to_device(args, group: Union[i_dist.Group, None]):
    if group is None:
        if context.global_strategy_type == tf.distribute.TPUStrategy:
            device_type = "TPU"
        else:
            device_type = "GPU"
        ranks = range(context.world_size)
    else:
        ranks = group.ranks
    for i, r in enumerate(ranks):
        with tf.device(f"{device_type}:{r}"):
            args[i] = args[i]
    return args


def pmap(
    fn: Callable,
    axis_name: Hashable,
    *,
    in_axes: Union[int, Sequence[int]] = 0,
    out_axes: Union[int, Sequence[int]] = 0,
    group: Union[i_dist.Group, None] = None,
    dst: int = 0,
):
    if isinstance(group, i_dist.Group):
        group_stratagy = group.ranks_to_torch_group()
        num_processes = len(group.ranks)
    else:
        group_stratagy = context.global_strategy
        num_processes = context.world_size

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if isinstance(in_axes, int):
            axes = (in_axes,) * len(args)

        if num_processes == 1:
            return ivy.vmap(fn, in_axes=in_axes, out_axes=out_axes)

        new_args = data_frag(*args, in_axes=axes, num_devices=num_processes)
        new_args = data_to_device(new_args, group=group)
        with group_stratagy.scope():
            replica_context = tf.distribute.get_replica_context()
            replica_id = replica_context.replica_id_in_sync_group
            func_out = ivy.vmap(fn, in_axes=in_axes, out_axes=out_axes)(
                *args[replica_id], **kwargs
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

    return wrapper
