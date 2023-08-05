from functools import wraps

import ivy.distributed as i_dist

context = i_dist.ParallelContext


def token_wrapper(fn):
    @wraps(fn)
    def _token_wrapper(*args, **kwargs):
        token = context.xla_token
        fn_out = fn(*args, **kwargs, token=token)
        if isinstance(fn_out, tuple):
            out, token = fn_out
        else:
            out = None
            token = fn_out
        context.xla_token = token
        return out

    return _token_wrapper
