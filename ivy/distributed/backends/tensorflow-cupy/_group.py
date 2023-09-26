import mpi4py.MPI as MPI
import cupyx.distributed as dist
import ivy.distributed as i_dist


class _CustomCupyGroup:
    def __init__(self, ranks):
        self.ranks = ranks
        pc = i_dist.ParallelContext()
        world_comm = MPI.COMM_WORLD
        try:
            self.g_rank = ranks.index(world_comm.Get_rank())
            self.in_group = True
        except ValueError:
            self.g_rank = None
            self.in_group = False
        if len(ranks) == MPI.COMM_WORLD.Get_size():
            self.comm = pc.world_comm
        elif self.g_rank is not None:
            if pc.backend == "mpi":
                self.comm = dist.init_process_group(
                    len(ranks), self.g_rank, backend=pc.backend, use_mpi=True
                )
            if pc.backend == "nccl":
                self.comm = dist.NCCLBackend(
                    len(ranks), self.g_rank, backend=pc.backend, use_mpi=False
                )
            else:
                raise Exception(
                    f"Tensorflow with cupy does not support the backend: {pc.backend}"
                )

    def __len__(self):
        return len(self.ranks)

    def all_gather(self, in_array, out_array, count, stream=None):
        self.comm.all_gather(in_array, out_array, count, stream)

    def all_reduce(self, in_array, out_array, op, stream=None):
        self.comm.all_reduce(in_array, out_array, op, stream)

    def all_to_all(self, in_array, out_array, stream=None):
        self.comm.all_to_all(in_array, out_array, stream)

    def barrier(self):
        self.comm.barrier()

    def broadcast(self, in_out_array, root, stream=None):
        self.comm.broadcast(in_out_array, root, stream)

    def gather(self, in_array, out_array, root, stream=None):
        self.comm.gather(in_array, out_array, root, stream)

    def recv(self, out_array, peer, stream=None):
        self.comm.recv(out_array, peer, stream)

    def reduce(self, in_array, out_array, root, op, stream=None):
        self.comm.reduce(in_array, out_array, root, op, stream)

    def reduce_scatter(self, in_array, out_array, count, op, stream=None):
        self.comm.reduce_scatter(in_array, out_array, count, op, stream)

    def scatter(self, in_array, out_array, root, stream=None):
        self.comm.scatter(in_array, out_array, root, stream)

    def send(self, array, peer, stream=None):
        self.comm.send(array, peer, stream)


def _to_native_group(ranks):
    i_dist.ParallelContext()
    return _CustomCupyGroup(ranks)


def _from_native_group(group: _CustomCupyGroup):
    return group.ranks


def _rank(group: _CustomCupyGroup):
    return group.g_rank
