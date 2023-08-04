import ivy


def init_dist(**kwargs) -> None:
    """
    Initialise the backend for distributed training. Different backends require
    different arguments, some backends dont require any arguments but still need to be
    initialised for example for jax each process should only be able to see a single
    cuda device, this is handled by calling ivy.disributed.init_dist()

    Parameters
    ----------
    backend : str
        This argument is exculsive to the torch backend,The backend to use. Depending
        on build-time configurations, valid values include mpi, gloo, nccl, and ucc. If
        the backend is not provied, then both a gloo and nccl backend will be created,
        see notes below for how multiple backends are managed. This field can be given
        as a lowercase string (e.g., "gloo"), which can also be accessed via Backend
        attributes (e.g., Backend.GLOO). If using multiple processes per machine with
        nccl backend, each process must have exclusive access to every GPU it uses, as
        sharing GPUs between processes can result in deadlocks. ucc backend is
        experimental.
    """
    ivy.current_dist_backend().init_dist(**kwargs)
