import torch
import torch.distributed as dist

import ivy
import ivy.distributed as i_dist

context = i_dist.ParallelContext


def all_reduce(x: torch.Tensor, op_handler: i_dist.OpHandler) -> torch.Tensor:
    op = i_dist.op_handler.torch_op
    work = dist.all_reduce(x, op, async_op=True)
    work.wait()
    return x


def all_gather(x: torch.Tensor, axis: int, tiled: bool = False):
    num_devices = context.world_size
    x = x if axis == 0 else x.transpose(0, axis).contiguous()
    out_shape = (x.shape[0] * num_devices,) + x.shape[1:]
    tensor_out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    work = dist.all_gather_into_tensor(tensor_out, x, group=None, async_op=True)
    work.wait()
    out = tensor_out if axis == 0 else tensor_out.transpose(0, axis)

    if tiled:
        out = ivy.split(out, num_or_size_splits=num_devices, axis=axis)

    return out


def all_to_all(x: torch.Tensor, axis: int) -> ivy.Array:
    out_shape = x.shape
    tensor_out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    work = dist.all_to_all(tensor_out, x, async_op=True)
    work.wait()
    return tensor_out
