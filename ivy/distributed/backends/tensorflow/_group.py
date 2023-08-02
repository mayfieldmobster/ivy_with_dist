import ivy
import ivy.distributed as i_dist
import tensorflow as tf


def to_native_group(ranks):
    context = i_dist.ParallelContext()

    if len(ranks) == context.world_size:
        return context.default_strategy

    if ivy.verbosity.level > 0:
        m = "Tensorflow only supports groups if strategy is MirroredStrategy\n"
        print(ivy.verbosity.cprint(m, color="red"))

    if context.global_strategy_type == tf.distribute.MirroredStrategy:
        devices = [f"GPU:{i}" for i in ranks]
        group = tf.distribute.MirroredStrategy(devices=devices)
    else:
        raise NotImplementedError(
            "Tensorflow groups only supported on Mirrored Strategy"
        )
    return group
