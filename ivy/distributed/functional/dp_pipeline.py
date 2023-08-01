from typing import Callable

import ivy.distributed as i_dist


def dp_pipeline(grad_fn: Callable, batch_axis: int = 0):
    def _dp_pipeline(*args, **kwargs):
        return i_dist.pmap(grad_fn, in_axes=batch_axis)(*args, **kwargs)

    return _dp_pipeline
