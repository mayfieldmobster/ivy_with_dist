import ivy.distributed as i_dist
import inspect


def group_handler(fn):
    signature = inspect.signature(fn)

    def _group_handler(*args, **kwargs):
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        context = i_dist.ParallelContext()
        if "group" in bound_args.arguments:
            if bound_args.arguments["group"] is None:
                bound_args.arguments["group"] = i_dist.Group(range(context.world_size))
            bound_args.arguments["group"] = bound_args.arguments[
                "group"
            ].to_native_group()
        return fn(*bound_args.args, **bound_args.kwargs)

    return _group_handler
