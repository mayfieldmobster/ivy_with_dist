import ivy.distributed as i_dist


def is_available():
    ...


def init_process_group(backend=None, **kwargs):
    i_dist.init_dist(backend=backend)


def is_initialized():
    return i_dist.is_initialized()


def is_mpi_available():
    ...


def is_nccl_available():
    ...


def is_gloo_available():
    ...


def is_torchelastic_launched():
    # must be launched via ivyrun
    return True
