class JaxGroupMixin:
    def ranks_to_jax_group(self):
        import mpi4py.MPI as MPI

        comm = MPI.COMM_WORLD
        group = comm.Get_group()

        new_group = group.excel(self.ranks)

        return comm.Create(new_group)
