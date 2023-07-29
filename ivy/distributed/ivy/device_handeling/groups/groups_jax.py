class JaxGroupMixin:
    def ranks_to_jax_devices(self):
        import jax

        devices = jax.devices()
        group_divices = []
        for i in self.ranks:
            group_divices.append(devices[i])
        return group_divices
