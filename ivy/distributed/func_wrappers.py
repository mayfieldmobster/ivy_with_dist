import ivy.distributed as i_dist


def group_none_handler(fn):
    def _group_handler(*args, **kwargs):
        context = i_dist.ParallelContext()
        for k, v in kwargs:
            if k == "group" and v is None:
                kwargs[k] = i_dist.Group(range(context.get_world_size))
        fn(*args, **kwargs)

    return _group_handler


def group_to_native(fn):
    def _group_to_native(*args, **kwargs):
        for k, v in kwargs:
            if k == "group":
                kwargs[k] == v.to_native_group()
                break
        fn(*args, **kwargs)

    return _group_to_native
