import os
from typing import Optional

import tensorflow as tf

import ivy.distributed as i_dist


def init_dist(
    world_size: int,
    multi_machine: bool,
    strategy: Optional[tf.distribute.Strategy] = None,
    cluster_resolver: Optional[tf.distribute.cluster_resolver.ClusterResolver] = None,
    **kwargs
):
    if multi_machine:
        try:
            os.environ["TF_CONFIG"]
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
                cluster_resolver=tf.distribute.TFConfigClusterResolver()
            )
            cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver
        except KeyError:
            if cluster_resolver is None:
                raise KeyError("TF_CONFIG must be defined")
            else:
                strategy = tf.distribute.MultiWorkerMirroredStrategy(
                    cluster_resolver=cluster_resolver
                )

    elif strategy is None:
        devices = tf.config.list_physical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy(devices=devices)

    context = i_dist.ParallelContext()
    context.reset_context()
    context.initilize()
    context.world_size = world_size
    context.multi_machine = multi_machine
    context.default_strategy_type = type(strategy)
    context.default_strategy = strategy
    context.device_type = (
        "GPU" if context.default_strategy_type != tf.distribute.TPUStrategy else "TPU"
    )
    context.default_cluster_resolver = cluster_resolver
    context.default_group = i_dist.Group(range(context.world_size))
