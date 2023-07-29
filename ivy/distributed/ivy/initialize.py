import ivy


def init_dist(
    world_size: int,
    multi_machine: bool,
    coordinator_address: str,
    **kwargs  # for framework_specific args
) -> None:
    if "tensorflow" in str(ivy.current_backend()):
        import tensorflow as tf

        for k, v in kwargs:
            if isinstance(v, tf.distribute.Strategy):
                break
        else:
            # TODO remove raw exception
            raise Exception(
                "When using Tensorflow backend tf.distribute.Strategy must be passed to"
                " ivy.dist.init_dist as an arg"
            )

    ivy.current_dist_backend().init_dist(
        world_size=world_size,
        multi_machine=multi_machine,
        coordinator_address=coordinator_address,
        **kwargs
    )
