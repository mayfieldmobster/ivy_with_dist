import mpi4py.MPI as MPI

import mpi4jax

from ivy.functional.backends.jax import JaxArray
from ._func_wrapper import token_wrapper


def send(x: JaxArray, dst: int, tag: int, group: MPI.Comm = MPI.COMM_WORLD):
    return token_wrapper(mpi4jax.send)(x=x, dest=dst, tag=tag, comm=group)


def recv(x_buffer: JaxArray, src: int, tag: int, group: MPI.Comm = MPI.COMM_WORLD):
    return token_wrapper(mpi4jax.recv)(x=x_buffer, source=src, tag=tag, comm=group)


def barrier(group: MPI.Comm = MPI.COMM_WORLD):
    token_wrapper(mpi4jax.barrier)(comm=group)
