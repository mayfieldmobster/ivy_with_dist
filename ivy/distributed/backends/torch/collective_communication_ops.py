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
    out=None,
) -> torch.Tensor:
    if isinstance(x, list):
        x = ivy.concat(x)
    op = op_handler.torch_op
    tensor_in = torch.clone(x).contiguous()
    dist.all_reduce(tensor_in, op, group=group)
    if out is not None:
        out[:] = tensor_in
        del tensor_in
    else:
        out = tensor_in
    if op_handler.name == "MEAN":
        out = out / group.size()
    return out


def all_gather(
    x: torch.Tensor,
    axis: int = 0,
    group: dist.ProcessGroup = dist.group.WORLD,
    tiled: bool = False,
    out=None,
):
    num_processes = group.size()
    tensor_in = x.contiguous() if axis == 0 else x.transpose(0, axis).contiguous()
    if out is None:
        tensor_out = [
            torch.empty(tensor_in.shape, dtype=tensor_in.dtype, device=tensor_in.device)
            for _ in range(num_processes)
        ]
    else:
        if isinstance(out, torch.Tensor):
            tensor_out = ivy.split(out, num_or_size_splits=num_processes)
        elif isinstance(out, list):
            tensor_out = out
        else:
            raise Exception("out must be list of tensors or tensor")

    dist.all_gather(tensor_out, tensor_in, group=group)
    tensor_out = ivy.concat(tensor_out)
    tensor_out = tensor_out if axis == 0 else tensor_out.transpose(0, axis)

    if tiled:
        tensor_out = ivy.split(tensor_out, num_or_size_splits=num_processes, axis=axis)

    return tensor_out


def all_to_all(
    x: torch.Tensor, group: dist.ProcessGroup = dist.group.WORLD, out=None
) -> ivy.Array:
    tensor_in = x.contiguous()
    out_shape = tensor_in.shape
    if out is None:
        tensor_out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    else:
        tensor_out = out
    dist.all_to_all_single(
        tensor_out,
        tensor_in,
        group=group,
    )
    return tensor_out


def broadcast(
    x: torch.Tensor, group: dist.ProcessGroup = dist.group.WORLD, src: int = 0
):
    tensor_in = x.contiguous()
    dist.broadcast(tensor=tensor_in, src=src, group=group)
    return tensor_in


def gather(
    x: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    axis: int = 0,
    tiled: bool = False,
    dst: int = 0,
    out=None,
):
    num_processes = group.size()
    # TODO add a method of getting process rank
    rank = group.rank()
    if num_processes == 1:
        return x
    tensor_in = x.contiguous() if axis == 0 else x.transpose(0, axis).contiguous()
    if dst == rank:
        if out is None:
            tensor_out = [
                torch.empty(
                    tensor_in.shape, dtype=tensor_in.dtype, device=tensor_in.device
                )
                for _ in range(num_processes)
            ]
        else:
            if isinstance(out, torch.Tensor):
                tensor_out = ivy.split(out, num_or_size_splits=num_processes)
            elif isinstance(out, list):
                tensor_out = out
            else:
                raise Exception("out must be list of tensors or tensor")
        dist.gather(tensor_in, tensor_out, dst=dst, group=group)
        tensor_out = ivy.concat(tensor_out)  # maybe change 0 to axis var
        tensor_out = tensor_out if axis == 0 else tensor_out.transpose(0, axis)
        if tiled:
            tensor_out = ivy.split(
                tensor_out, num_or_size_splits=num_processes, axis=axis
            )
        return tensor_out
    else:
        dist.gather(tensor_in, dst=dst, group=group)
        return True


def reduce(
    x: torch.Tensor,
    op_handler: i_dist.OpHandler,
    group: dist.ProcessGroup = dist.group.WORLD,
    dst: int = 0,
    out=None,
):
    tensor_in = torch.clone(x).contiguous()
    op = op_handler.torch_op
    dist.reduce(tensor_in, dst=dst, op=op, group=group)
    if group.rank() == dst:
        if out is not None:
            out[:] = tensor_in
            del tensor_in
        else:
            out = tensor_in
        if op_handler.name == "MEAN":
            out = out / group.size()
    else:
        out = None
    return out


def scatter(
    out_buffer: torch.Tensor,
    x: torch.Tensor = None,
    group: dist.ProcessGroup = dist.group.WORLD,
    src: int = 0,
):
    if group.rank() == src:
        tensor_in = x.contiguous()
        tensor_in = list(torch.chunk(tensor_in, group.size()))
    else:
        tensor_in = x
    dist.scatter(tensor=out_buffer, scatter_list=tensor_in, src=src, group=group)
    return out_buffer


def reduce_scatter(
    x: torch.Tensor,
    op_handler: i_dist.OpHandler,
    group: dist.ProcessGroup = dist.group.WORLD,
    out=None,
):
    tensor_out = []
    num_processes = group.size()
    x = ivy.split(x, num_or_size_splits=num_processes)
    outs = [None] * num_processes
    outs[group.rank()] = out
    for dst, tensor_in in enumerate(x):
        tensor_out.append(
            reduce(
                tensor_in, op_handler=op_handler, group=group, dst=dst, out=outs[dst]
            )
        )

    i_dist.barrier(group=group)
    return tensor_out[group.rank()]
