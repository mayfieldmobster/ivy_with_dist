import mpi4py.MPI as MPI

import ivy.distributed as i_dist


pc = i_dist.ParallelContext()


def comm_to_group(fn):
    # comm will be passed as an MPI.Comm but i_dist funcs support i_dist.Group
    # this will depreciate once the mpi4py frontend is complete
    def _comm_to_group(*args, **kwargs):
        comm = kwargs["comm"]
        w_group = MPI.COMM_WORLD.Get_group()
        group = comm.Get_group()
        group = i_dist.Group(
            group.Translate_ranks(w_group, range(MPI.COMM_WORLD.Get_size()))
        )
        kwargs["comm"] = group
        return fn(*args, **kwargs)

    return _comm_to_group


def _op_to_ophandler(op):
    if op == MPI.SUM:
        return i_dist.OpHandler("SUM")
    elif op == MPI.MAX:
        return i_dist.OpHandler("MAX")
    elif op == MPI.MIN:
        return i_dist.OpHandler("MIN")
    else:
        raise Exception("Op has not been implemented yet")


@comm_to_group
def allgather(x, *, comm=None, token=None):
    out = i_dist.all_gather(x=x, group=comm)
    if hasattr(pc, "token"):
        return out, pc.token
    else:
        return out, token


@comm_to_group
def allreduce(x, op, *, comm=None, token=None):
    op = _op_to_ophandler(op)
    out = i_dist.all_reduce(x=x, op=op, group=comm)
    if hasattr(pc, "token"):
        return out, pc.token
    else:
        return out, token


@comm_to_group
def alltoall(x, *, comm=None, token=None):
    out = i_dist.all_to_all(x, group=comm)
    if hasattr(pc, "token"):
        return out, pc.token
    else:
        return out, token


@comm_to_group
def barrier(*, comm=None, token=None):
    i_dist.barrier(group=comm)
    if hasattr(pc, "token"):
        return pc.token
    else:
        return token


@comm_to_group
def bcast(x, root, *, comm=None, token=None):
    out = i_dist.broadcast(x=x, src=root, group=comm)
    if hasattr(pc, "token"):
        return out, pc.token
    else:
        return out, token


@comm_to_group
def gather(x, root, *, comm=None, token=None):
    out = i_dist.gather(x=x, axis=0, group=comm, tiled=False, dst=root)
    if hasattr(pc, "token"):
        return out, pc.token
    else:
        return out, token


@comm_to_group
def reduce(x, op, root, *, comm=None, token=None):
    op = _op_to_ophandler(op)
    out = i_dist.reduce(x=x, op=op, group=comm, dst=root)
    if hasattr(pc, "token"):
        return out, pc.token
    else:
        return out, token


@comm_to_group
def scan(x, op, *, comm=None, token=None):
    ...


@comm_to_group
def scatter(x, root, *, comm=None, token=None):
    if pc.rank == 0:
        out = x[0]
    else:
        out = x
    out = i_dist.scatter(out, x, group=comm, src=root)
    if hasattr(pc, "token"):
        return out, pc.token
    else:
        return out, token
