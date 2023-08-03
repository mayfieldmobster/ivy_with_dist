import os


def get_global_rank():
    return int(os.environ["RANK"])


def get_local_rank():
    return int(os.environ["LOCAL_RANK"])


def get_world_size():
    return int(os.environ["WORLD_SIZE"])
