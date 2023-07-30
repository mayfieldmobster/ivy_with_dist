from typing import Optional, Union, Sequence

import jax

import ivy.distributed as i_dist


def init_dist(
    world_size: int,
    multi_machine: bool,
    coordinator_address: Optional[str] = None,
    process_id: Optional[int] = None,
    local_device_ids: Optional[Union[int, Sequence[int]]] = None,
    **kwargs
):
    if multi_machine:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=world_size,
            process_id=process_id,
            local_device_ids=local_device_ids,
        )

    context = i_dist.ParallelContext()
    context.reset_context()
    context.world_size = world_size
    context.multi_machine = multi_machine
