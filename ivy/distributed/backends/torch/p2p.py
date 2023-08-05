import torch
import torch.distributed as dist


def send(
    x: torch.Tensor, dst: int, tag: int, group: dist.ProcessGroup = dist.group.WORLD
):
    return dist.send(tensor=x, dst=dst, group=group, tag=tag)


def recv(
    x_buffer: torch.Tensor,
    src: int,
    tag: int,
    group: dist.ProcessGroup = dist.group.WORLD,
):
    if x_buffer.numel() != 0:
        x_buffer = torch.empty_like(x_buffer)
    dist.recv(x_buffer, src=src, group=group, tag=tag)
    return x_buffer


def barrier(group: dist.ProcessGroup = dist.group.WORLD):
    dist.barrier(group=group)
