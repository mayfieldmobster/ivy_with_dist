import mpi4py.MPI as MPI
import os

import mpi4jax

import ivy.distributed as i_dist


def init_dist(**kwargs):
    mpi_comm = MPI.COMM_WORLD
    mpi_comm.Get_size()
    local_rank = MPI.COMM_WORLD.Get_rank() % os.environ["NPROC_PER_NODE"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    token = mpi4jax.barrier()

    context = i_dist.ParallelContext()
    context.reset_context()
    context.initilize()
    multi_machine = True if context.world_size / context.rank != 1 else False
    context.multi_machine = multi_machine
    context.xla_token = token
    context.default_group = i_dist.Group(range(context.world_size))
