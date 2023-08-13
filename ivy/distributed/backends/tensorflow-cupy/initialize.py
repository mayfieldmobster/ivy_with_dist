import cupyx.distributed as dist
import mpi4py.MPI as MPI

import ivy.distributed as i_dist


def init_dist(backend="mpi", **kwargs):
    context = i_dist.ParallelContext()
    context.reset_context()
    context.initilize()
    if backend == "mpi":
        comm = dist.init_process_group(
            MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank(), use_mpi=True
        )
    elif backend == "nccl":
        comm = dist.NCCLBackend(
            MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank(), use_mpi=True
        )
    context.world_comm = comm
    context.default_group = i_dist.Group(range(context.world_size))
