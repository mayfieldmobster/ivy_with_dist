import jax
import jax.lax as lax

import ivy.distributed as i_dist
from ivy.functional.backends.jax import JaxArray


def all_reduce(
    x: JaxArray, op_handler: i_dist.OpHandler, group: i_dist.Group = None
) -> JaxArray:
    # TODO support inplace all reduce
    x = all_gather(x, axis=0, group=group, tiled=True)
    op = op_handler.jax_op
    x = op(x)
    return x


def all_gather(
    x: JaxArray, axis: int - 0, group: i_dist.Group = None, tiled: bool = False
) -> JaxArray:
    devices = group.ranks_to_jax_devices
    # random axis name to prevent error
    f = lambda x: lax.all_gather(x, axis_name="zgfjklk", tiled=tiled)
    return jax.pmap(f, in_axis=axis, axis_name="zgfjklk", devices=devices)(x)


def all_to_all(x: JaxArray, axis: int = 0, group: i_dist.Group = None) -> JaxArray:
    ...


def gather(
    x: JaxArray,
    axis: int = 0,
    group: i_dist.Group = None,
    tiled: bool = False,
    dst: int = 0,
):
    ...


def reduce(
    x: JaxArray,
    op: i_dist.OpHandler,
    group: i_dist.Group = None,
    dst: int = 0,
):
    ...
