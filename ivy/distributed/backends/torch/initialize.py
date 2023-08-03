from typing import Optional

import torch.distributed as dist

import ivy.distributed as i_dist


def init_dist(
    backend: Optional[str] = None,
    **kwargs,
):
    dist.init_process_group(backend=backend)

    context = i_dist.ParallelContext()
    context.reset_context()
    context.initilize()

    multi_machine = True if context.world_size / context.rank != 1 else False
    context.multi_machine = multi_machine
    context.default_group = i_dist.Group(range(context.world_size))
