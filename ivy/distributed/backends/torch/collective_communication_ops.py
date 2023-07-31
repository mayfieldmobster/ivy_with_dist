from typing import Union

import torch
import torch.distributed as dist

import ivy
import ivy.distributed as i_dist

context = i_dist.ParallelContext


# TODO add a scalar to vector call of x to check


def all_reduce(
    x: torch.Tensor, op_handler: i_dist.OpHandler, group: i_dist.Group = None
) -> torch.Tensor:
    op = i_dist.op_handler.torch_op
    group = group.ranks_to_torch_group()
    work = dist.all_reduce(x, op, group=group, async_op=True)
    work.wait()
    return x


def all_gather(
    x: torch.Tensor, axis: int = 0, group: i_dist.Group = None, tiled: bool = False
):
    group = group.ranks_to_torch_group()
    num_processes = group.size
    x = x if axis == 0 else x.transpose(0, axis).contiguous()
    out_shape = (x.shape[0] * num_processes,) + x.shape[1:]
    tensor_out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    work = dist.all_gather_into_tensor(tensor_out, x, group=None, async_op=True)
    work.wait()
    out = tensor_out if axis == 0 else tensor_out.transpose(0, axis)

    if tiled:
        out = ivy.split(out, num_or_size_splits=num_processes, axis=axis)

    return out


def all_to_all(
    x: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group: i_dist.Group = None,
) -> ivy.Array:
    input_tensor = x.contiguous()
    out_shape = input_tensor.shape
    tensor_out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    work = dist.all_to_all(
        tensor_out,
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True,
    )
    work.wait()
    return tensor_out


def gather(
    x: torch.Tensor,
    group: Union[i_dist.Group, dist.ProcessGroup],
    axis: int = 0,
    tiled: bool = False,
    dst: int = 0,
):
    group = group.ranks_to_torch_group()
    num_processes = group.size()
    # TODO add a method of getting process rank
    rank = dist.get_group_rank(group, int)
    if num_processes == 1:
        out = x
    tensor_in = x.contiguous() if axis == 0 else x.transpose(0, axis)

    if dst == rank:
        tensor_out = [
            torch.empty(tensor_in.shape, device=tensor_in.device, dtype=tensor_in.dtype)
            for _ in range(num_processes)
        ]
        work = dist.gather(tensor_in, tensor_out, dst=dst, group=group, async_op=True)
        tensor_out = torch.cat(tensor_out, dim=0)  # maybe change 0 to axis var
        out = tensor_out if axis == 0 else tensor_out.transpose(0, axis)
        work.wait()
        if tiled:
            out = ivy.split(out, num_or_size_splits=num_processes, axis=axis)
        return out
    else:
        work = dist.gather(tensor_in, dst=dst, group=group, async_op=True)
        work.wait()
        return True
