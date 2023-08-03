import mpi4py.MPI as MPI


def _to_native_group(ranks):
    comm = MPI.COMM_WORLD

    if len(ranks) == comm.Get_size():
        return comm

    group = comm.Get_group()

    new_group = group.excel(ranks)

    return comm.Create(new_group)
