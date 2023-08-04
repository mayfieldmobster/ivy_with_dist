import os
import json


def get_global_rank():
    ...


def get_local_rank():
    ...


def get_world_size():
    config = json.loads(os.environ["TF_CONFIG"])
    return len(config["cluster"]["worker"])
