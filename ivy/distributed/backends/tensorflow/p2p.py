import mpi4py.MPI as MPI

import cupy as cp

from _func_wrapper import to_dlpack_and_back


@to_dlpack_and_back
def send(x: cp.ndarray, dst: int, tag: int, group: MPI.Comm = MPI.COMM_WORLD):
    group.Send(x, dest=dst, tag=tag)


@to_dlpack_and_back
def recv(x_buffer: cp.ndarray, src: int, tag: int, group: MPI.Comm = MPI.COMM_WORLD):
    out = cp.empty_like(x_buffer, dtype=x_buffer.dtype)
    group.Recv(out, source=src, tag=tag)
    return out
