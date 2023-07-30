import tensorflow as tf

import ivy
import ivy.distributed as i_dist


class TFGroupMixin:
    def ranks_to_tf_group(self):
        context = i_dist.ParallelContext()

        if ivy.verbosity.level > 0:
            m = "Tensorflow only supports groups if strategy is MirroredStrategy\n"
            print(ivy.verbosity.cprint(m, color="red"))

        if context.global_strategy_type == tf.distribute.MirroredStrategy:
            devices = [f"GPU:{i}" for i in self.ranks]
            group = tf.distribute.MirroredStrategy(devices=devices)
        else:
            raise NotImplementedError(
                "Tensorflow groups only supported on Mirrored Strategy"
            )
        return group
