import mpi4py.MPI as MPI

import mpi4jax

from ivy.functional.backends.jax import JaxArray
from ._func_wrapper import token_wrapper


def send(x: JaxArray, dst: int, tag: int, group: MPI.Comm = MPI.COMM_WORLD):
    comm = group
    return token_wrapper(mpi4jax.send)(x=x, dest=dst, tag=tag, comm=comm)


def recv(x_buffer: JaxArray, src: int, tag: int, group: MPI.Comm = MPI.COMM_WORLD):
    comm = group
    return token_wrapper(mpi4jax.recv)(x=x_buffer, source=src, tag=tag, comm=comm)
