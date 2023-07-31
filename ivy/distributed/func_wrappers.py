import ivy.distributed as i_dist


def group_handler(fn):
    def _group_handler(*args, **kwargs):
        context = i_dist.ParallelContext()
        for k, v in kwargs:
            if k == "group" and v is None:
                kwargs[k] = context.default_group

        fn(*args, **kwargs)

    return _group_handler
