from typing import Union

import tensorflow as tf

# from tensorflow.compiler.tf2xla.python import xla

import ivy.distributed as i_dist

context = i_dist.ParallelContext()

# TODO implement tf.raw_ops.Send/Recv


def send(x: Union[tf.Variable, tf.Tensor], dst: int, tag: int, group: i_dist.Group):
    global dic
    with tf.device(f"{context.device_type}:{group[dst]}"):
        dic[tag] = x


def recv(
    x_buffer: Union[tf.Variable, tf.Tensor], src: int, tag: int, group: i_dist.Group
):
    global dic
    return dic.pop(tag)
