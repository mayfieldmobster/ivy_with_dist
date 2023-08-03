import ivy


def init_dist(
    multi_machine: bool = False,
    coordinator_address: str = None,
    **kwargs  # for framework_specific args
) -> None:
    # TODO check if code is necessary
    """
    If ivy.backend == "tensorflow": import tensorflow as tf.

    for k, v in kwargs:
        if isinstance(v, tf.distribute.Strategy):
            break
    else:
        # TODO remove raw exception
        raise Exception(
            "When using Tensorflow backend tf.distribute.Strategy must be passed to"
            " ivy.dist.init_dist as an arg"
        )
    """
    ivy.current_dist_backend().init_dist(
        multi_machine=multi_machine, coordinator_address=coordinator_address, **kwargs
    )
