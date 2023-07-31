from functools import wraps

import ivy.distributed as i_dist

context = i_dist.ParallelContext


def token_wrapper(fn):
    @wraps(fn)
    def _token_wrapper(*args, **kwargs):
        token = context.xla_token
        out, token = fn(*args, token=token, **kwargs)
        context.xla_token = token
        return out

    return _token_wrapper
