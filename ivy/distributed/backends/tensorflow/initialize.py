import os

import tensorflow as tf

import ivy.distributed as i_dist


def init_dist(
    world_size: int, multi_machine: bool, strategy: tf.distribute.Strategy, **kwargs
):
    if multi_machine:
        try:
            os.environ["TF_CONFIG"]
        except KeyError:
            raise KeyError("TF_CONFIG must be defined")

    context = i_dist.ParallelContext()
    context.reset_context()
    context.world_size = world_size
    context.strategy_type = type(strategy)
    context.global_stratagy = strategy
