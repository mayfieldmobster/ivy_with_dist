from typing import Optional

import ivy.distributed as i_dist


def init_dist(
    multi_machine: bool,
    coordinator_address: Optional[str] = None,
    shared_file_system_path: Optional[str] = None,
    backend: Optional[str] = None,
):
    context = i_dist.ParallelContext()
    context.reset_context()
    context.initilize()
    context.multi_machine = multi_machine
    context.default_group = i_dist.Group(range(context.world_size))
