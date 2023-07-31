import ivy
import ivy.distributed as i_dist


# TODO explore nested stratagies
# ```python
# import tensorflow as tf
# node_strat = tf.distribute.MultiWorkerMirroredStrategy
# with node_strat.scope():
#    gpu_strat = tf.distrubute.MirroredStrategy
#    with gpu_strat.scope():
#        global_rank = node_strat.replica_id_in_sync_group
#        local_rank = gpu_strat.replica_id_in_sync_group
# ```


class TFGroupMixin:
    def ranks_to_tf_group(self):
        import tensorflow as tf

        context = i_dist.ParallelContext()

        if len(self.ranks) == context.world_size:
            return context.default_strategy

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
