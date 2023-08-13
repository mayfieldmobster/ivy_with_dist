import mpi4py.MPI as MPI
import os


def get_global_rank():
    return MPI.COMM_WORLD.Get_rank()


def get_local_rank():
    # not sure if this works
    return MPI.COMM_WORLD.Get_rank() % os.environ["NPROC_PER_NODE"]


def get_world_size():
    return MPI.COMM_WORLD.Get_size()
