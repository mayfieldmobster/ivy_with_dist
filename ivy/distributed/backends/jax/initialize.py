from typing import Optional, Union, Sequence
import mpi4py.MPI as MPI
import os

import mpi4jax

import ivy.distributed as i_dist


def init_dist(
    multi_machine: bool,
    coordinator_address: Optional[str] = None,
    process_id: Optional[int] = None,
    local_device_ids: Optional[Union[int, Sequence[int]]] = None,
    **kwargs
):
    mpi_comm = MPI.COMM_WORLD
    mpi_comm.Get_size()
    local_rank = MPI.COMM_WORLD.Get_rank() % os.environ["NPROC_PER_NODE"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    token = mpi4jax.barrier()

    context = i_dist.ParallelContext()
    context.reset_context()
    context.initilize()
    context.multi_machine = multi_machine
    context.xla_token = token
    context.default_group = i_dist.Group(range(context.world_size))
