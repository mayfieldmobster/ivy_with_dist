from .groups_torch import TorchGroupMixin
from .groups_jax import JaxGroupMixin
from .groups_tf import TFGroupMixin


class Group(TorchGroupMixin, JaxGroupMixin, TFGroupMixin):
    def __init__(self, ranks):
        self.ranks = ranks
