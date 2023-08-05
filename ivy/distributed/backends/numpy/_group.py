import mpi4py.MPI as MPI


def _to_native_group(ranks):
    comm = MPI.COMM_WORLD

    if ranks == list(range(comm.Get_size())):
        return comm

    group = comm.Get_group()

    new_group = group.Incl(ranks)

    return comm.Create(new_group)


def _from_native_group(comm: MPI.Comm):
    w_group = MPI.COMM_WORLD.Get_group()
    group = comm.Get_group()
    return group.Translate_ranks(w_group, range(MPI.COMM_WORLD.Get_size()))


def _rank(comm: MPI.Comm):
    return comm.Get_rank()
