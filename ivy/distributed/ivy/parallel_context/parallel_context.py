from ivy.utils.singleton import SingletonMetaClass
from ivy import current_dist_backend


class ParallelContext(dict, metaclass=SingletonMetaClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self._is_initized = False

    def initilize(self):
        self._is_initized = True

    @property
    def is_initized(self):
        return self._is_initized

    def reset_context(self):
        self.clear()

    @property
    def rank(self):
        return self.get_global_rank()

    @property
    def local_rank(self):
        return self.get_local_rank()

    @property
    def world_size(self):
        return self.get_world_size()

    def get_global_rank(self):
        return current_dist_backend()._context.get_global_rank()

    def get_local_rank(self):
        return current_dist_backend()._context.get_local_rank()

    def get_world_size(self):
        return current_dist_backend()._context.get_world_size()
