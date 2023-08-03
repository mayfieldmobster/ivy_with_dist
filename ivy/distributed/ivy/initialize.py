import ivy


def init_dist(**kwargs) -> None:  # for framework_specific args
    ivy.current_dist_backend().init_dist(**kwargs)
