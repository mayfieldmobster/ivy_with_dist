from abc import ABC, abstractmethod

from ivy.utils.singleton import SingletonMetaClass


class ParallelContext(ABC, metaclass=SingletonMetaClass):
    def __init__(self, world_size):
        super().__init__()
        self.world_size = world_size
        self.process_groups = dict()

    @abstractmethod
    def get_world_size():
        pass
