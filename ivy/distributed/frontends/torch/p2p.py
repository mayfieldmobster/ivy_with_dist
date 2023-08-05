import ivy.distributed as i_dist


def send(tensor, dst, group=None, tag=0):
    i_dist.send(x=tensor, dst=dst, group=group, tag=tag)


def recv(tensor, src, group=None, tag=0):
    return i_dist.recv(x_buffer=tensor, src=src, group=group, tag=tag)


def isend(tensor, dst, group=None, tag=0):
    raise Exception("Asynchronous Implementations not supported yet")


def irecv(tensor, src=None, group=None, tag=0):
    raise Exception("Asynchronous Implementations not supported yet")


def batch_isend_irecv(p2p_op_list):
    raise Exception("Asynchronous Implementations not supported yet")
