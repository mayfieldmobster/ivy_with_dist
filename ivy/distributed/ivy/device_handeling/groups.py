import ivy


class Group:
    def __init__(self, ranks):
        self.ranks = ranks

    def __len__(self):
        return len(self.ranks)

    @property
    def size(self):
        return self.__len__()

    def __getitem__(self, idx):
        return self.ranks[idx]

    def to_native_group(self):
        return ivy.current_dist_backend()._group._to_native_group(self.ranks)
