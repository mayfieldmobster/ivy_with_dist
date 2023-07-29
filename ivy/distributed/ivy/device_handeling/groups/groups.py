from .groups_torch import TorchGroupMixin
from .groups_jax import JaxGroupMixin


class Group(TorchGroupMixin, JaxGroupMixin):
    def __init__(self, ranks):
        self.ranks = ranks
