import ivy
from typing import Sequence


class Group:
    def __init__(self, group):
        if isinstance(group, Sequence):
            self.ranks = list(group)
        else:
            self.ranks = self.from_native_group(group)

    def __len__(self):
        return len(self.ranks)

    @property
    def size(self):
        return self.__len__()

    def __getitem__(self, idx):
        return self.ranks[idx]

    def to_native_group(self):
        return ivy.current_dist_backend()._group._to_native_group(self.ranks)

    def from_native_group(self, native_group):
        self.rank = ivy.current_dist_backend()._group._from_native_group(native_group)
