import ivy.distributed as i_dist


pc = i_dist.ParallelContext()


def recv(x, source=-1, *, tag=-1, comm=None, status=None, token=None):
    out = i_dist.recv(x_buffer=x, src=source, group=comm, tag=tag)
    if hasattr(pc, "token"):
        return out, pc.token
    else:
        return out, token


def send(x, dest, *, tag=0, comm=None, token=None):
    i_dist.send(x=x, dst=dest, group=comm, tag=tag)
    if hasattr(pc, "token"):
        return pc.token
    else:
        return token


def sendrecv(
    sendbuf,
    recvbuf,
    source,
    dest,
    *,
    sendtag=0,
    recvtag=-1,
    comm=None,
    status=None,
    token=None
):
    ...
