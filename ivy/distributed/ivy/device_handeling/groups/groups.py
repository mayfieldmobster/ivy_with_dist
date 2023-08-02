from .groups_torch import TorchGroupMixin
from .groups_jax import JaxGroupMixin
from .groups_tf import TFGroupMixin
from .groups_numpy import NumpyGroupMixin


class Group(TorchGroupMixin, JaxGroupMixin, TFGroupMixin, NumpyGroupMixin):
    def __init__(self, ranks):
        self.ranks = ranks

    def __len__(self):
        return len(self.ranks)

    @property
    def size(self):
        return self.__len__()

    def __getitem__(self, idx):
        return self.ranks[idx]
