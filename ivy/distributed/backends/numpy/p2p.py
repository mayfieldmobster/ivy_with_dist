import mpi4py.MPI as MPI

import numpy as np


def send(x: np.ndarray, dst: int, tag: int, group: MPI.Comm = MPI.COMM_WORLD):
    group.Send(x, dest=dst, tag=tag)


def recv(x_buffer: np.ndarray, src: int, tag: int, group: MPI.Comm = MPI.COMM_WORLD):
    out = np.empty_like(x_buffer)
    out = group.Recv(x_buffer, source=src, tag=tag)
    return out
