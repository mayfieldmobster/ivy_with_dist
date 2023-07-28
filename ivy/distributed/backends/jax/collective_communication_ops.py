import jax
import jax.lax as lax

import ivy.distributed as i_dist
from ivy.functional.backends.jax import JaxArray


def all_reduce(x: JaxArray, op_handler: i_dist.OpHandler) -> JaxArray:
    # TODO support inplace all reduce
    x = all_gather(x, axis=0)
    op = op_handler.jax_op
    x = op(x)
    return x


def all_gather(x: JaxArray, axis: int, tiled: bool = False) -> JaxArray:
    # random axis name to prevent error
    f = lambda x: lax.all_gather(x, axis_name="zgfjk", tiled=tiled)
    out = jax.pmap(f, in_axis=axis, axis_name="zgfjk")
    return out


def all_to_all(x: JaxArray, axis: int) -> JaxArray:
    ...
