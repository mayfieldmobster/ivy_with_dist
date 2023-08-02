from typing import Optional

import torch.distributed as dist

import ivy.distributed as i_dist


def init_dist(
    world_size: int,
    multi_machine: bool,
    coordinator_address: Optional[str] = None,
    shared_file_system_path: Optional[str] = None,
    backend: Optional[str] = None,
    **kwargs,
):
    if multi_machine:
        if shared_file_system_path:
            init_method = f"file://{shared_file_system_path}"
        else:
            init_method = f"tcp://{coordinator_address}"
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
        )

    context = i_dist.ParallelContext()
    context.reset_context()
    context.initilize()
    context.world_size = world_size
    context.multi_machine = multi_machine
    context.default_group = i_dist.Group(range(context.world_size))
