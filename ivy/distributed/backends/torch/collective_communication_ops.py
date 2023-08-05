import torch
import torch.distributed as dist

import ivy
import ivy.distributed as i_dist

context = i_dist.ParallelContext


# TODO add a scalar to vector call of x to check


def all_reduce(
    x: torch.Tensor,
    op_handler: i_dist.OpHandler,
    group: dist.ProcessGroup = dist.group.WORLD,
) -> torch.Tensor:
    op = op_handler.torch_op
    tensor_in = x.contiguous()
    work = dist.all_reduce(tensor_in, op, group=group, async_op=True)
    work.wait()
    if op_handler.name == "MEAN":
        tensor_in = tensor_in / group.size
    return tensor_in


def all_gather(
    x: torch.Tensor,
    axis: int = 0,
    group: dist.ProcessGroup = dist.group.WORLD,
    tiled: bool = False,
):
    num_processes = group.size
    tensor_in = x.contiguous() if axis == 0 else x.transpose(0, axis).contiguous()
    tensor_out = [
        torch.empty(tensor_in.shape, dtype=tensor_in.dtype, device=tensor_in.device)
        for _ in range(group.size())
    ]
    work = dist.all_gather(tensor_out, tensor_in, group=group, async_op=True)
    work.wait()
    tensor_out = ivy.concat(tensor_out)
    tensor_out = tensor_out if axis == 0 else tensor_out.transpose(0, axis)

    if tiled:
        tensor_out = ivy.split(tensor_out, num_or_size_splits=num_processes, axis=axis)

    return tensor_out


def all_to_all(
    x: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group: dist.ProcessGroup = dist.group.WORLD,
) -> ivy.Array:
    tensor_in = x.contiguous()
    out_shape = tensor_in.shape
    tensor_out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    work = dist.all_to_all(
        tensor_out,
        tensor_in,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True,
    )
    work.wait()
    return tensor_out


def broadcast(
    x: torch.Tensor, group: dist.ProcessGroup = dist.group.WORLD, src: int = 0
):
    tensor_in = x.contiguous()
    work = dist.broadcast(tensor=tensor_in, src=src, group=group, async_op=True)
    work.wait()
    return tensor_in


def gather(
    x: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    axis: int = 0,
    tiled: bool = False,
    dst: int = 0,
):
    num_processes = group.size()
    # TODO add a method of getting process rank
    rank = group.rank()
    if num_processes == 1:
        return x
    tensor_in = x.contiguous() if axis == 0 else x.transpose(0, axis).contiguous()

    if dst == rank:
        tensor_out = [
            torch.empty(x.shape, device=tensor_in.device, dtype=tensor_in.dtype)
            for _ in range(group.size())
        ]
        work = dist.gather(tensor_in, tensor_out, dst=dst, group=group, async_op=True)
        tensor_out = ivy.concat(tensor_out)  # maybe change 0 to axis var
        tensor_out = tensor_out if axis == 0 else tensor_out.transpose(0, axis)
        work.wait()
        if tiled:
            tensor_out = ivy.split(
                tensor_out, num_or_size_splits=num_processes, axis=axis
            )
        return tensor_out
    else:
        work = dist.gather(tensor_in, dst=dst, group=group, async_op=True)
        work.wait()
        return True


def reduce(
    x: torch.Tensor,
    op_handler: i_dist.OpHandler,
    group: dist.ProcessGroup = dist.group.WORLD,
    dst: int = 0,
):
    tensor_in = x.contiguous()
    op = op_handler.torch_op
    work = dist.reduce(tensor_in, dst=dst, op=op, group=group, async_op=True)
    work.wait()
    if op_handler.name == "MEAN":
        tensor_in = tensor_in / group.size
    return tensor_in


def scatter(
    out_buffer: torch.Tensor,
    x: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    src: int = 0,
):
    tensor_in = x.contiguous()
    tensor_in = list(torch.chunk(tensor_in, group.size()))
    work = dist.scatter(
        tensor=out_buffer, scatter_list=tensor_in, src=src, group=group, async_op=True
    )
    work.wait()
    return out_buffer


def reduce_scatter(
    x: torch.Tensor,
    op_handler: i_dist.OpHandler,
    group: dist.ProcessGroup = dist.group.WORLD,
):
    x = ivy.split(x, num_or_size_splits=group.size())
    for dst, tensor_in in enumerate(x):
        out = reduce(tensor_in, op_handler=op_handler, group=group, dst=dst)

    i_dist.barrier(group=group)
    return out
