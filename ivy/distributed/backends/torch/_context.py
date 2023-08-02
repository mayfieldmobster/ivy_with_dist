import os


def get_global_rank():
    return os.environ["RANK"]


def get_local_rank():
    return os.environ["LOCAL_RANK"]


def get_world_size():
    return os.environ["WORLD_SIZE"]
