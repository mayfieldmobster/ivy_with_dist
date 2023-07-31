from ivy.utils.singleton import SingletonMetaClass


class ParallelContext(dict, metaclass=SingletonMetaClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def reset_context(self):
        self.clear()

    def get_global_rank(self):
        ...
