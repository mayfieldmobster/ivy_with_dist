from typing import Optional, Union, Sequence
import mpi4py.MPI as MPI
import os

import jax
import mpi4jax

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
    mpi_comm = MPI.COMM_WORLD
    global_rank = mpi_comm.Get_rank()
    world_size = mpi_comm.Get_size()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(global_rank)

    token = mpi4jax.barrier()

    context = i_dist.ParallelContext()
    context.reset_context()
    context.world_size = world_size
    context.multi_machine = multi_machine
    context.global_rank = global_rank
    context.xla_token = token
    context.default_group = i_dist.Group(range(context.world_size))
