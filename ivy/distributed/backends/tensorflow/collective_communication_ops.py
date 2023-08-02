from typing import Union

import tensorflow as tf

import ivy.distributed as i_dist
import ivy

context = i_dist.ParallelContext()

# TODO change to work with to_native_group


def _to_all_devices(x: Union[tf.Variable, tf.Tensor]):
    stratagy: tf.distribute.Strategy = context.global_stratagy
    with stratagy.scope():
        x = x
    return x


def all_reduce(
    x: Union[tf.Variable, tf.Tensor],
    op_handler: i_dist.OpHandler,
    group: i_dist.Group = None,
) -> Union[tf.Variable, tf.Tensor]:
    stratagy = group.ranks_to_tf_group
    reduced_x = stratagy.reduce(op_handler.tensorflow_op, x)
    return _to_all_devices(reduced_x)


def all_gather(
    x: Union[tf.Variable, tf.Tensor],
    axis: int = 0,
    group: i_dist.Group = None,
    tiled: bool = False,
) -> Union[tf.Variable, tf.Tensor]:
    num_devices = len(group)
    stratagy = group.ranks_to_tf_group
    gathered_x = stratagy.gather(x, axis)
    if tiled:
        gathered_x = ivy.split(gathered_x, num_or_size_splits=num_devices, axis=axis)

    return _to_all_devices(gathered_x)


def all_to_all(
    x: Union[tf.Variable, tf.Tensor], axis: int = 0, group: i_dist.Group = None
) -> Union[tf.Variable, tf.Tensor]:
    ...


def gather(
    x: Union[tf.Variable, tf.Tensor],
    axis: int = 0,
    group: i_dist.Group = None,
    tiled: bool = False,
    dst: int = 0,
):
    num_devices = len(group)
    stratagy = group.ranks_to_tf_group
    gathered_x = stratagy.gather(x, axis)
    if tiled:
        gathered_x = ivy.split(gathered_x, num_or_size_splits=num_devices, axis=axis)
    for i in range(len(group)):
        with tf.device(f"GPU:{group[i]}"):
            if i == dst:
                out = gathered_x
            else:
                out = True
    return out
