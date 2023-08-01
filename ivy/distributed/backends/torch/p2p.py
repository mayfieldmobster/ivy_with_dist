import torch
import torch.distributed as dist

import ivy.distributed as i_dist


def send(x: torch.Tensor, dst: int, tag: int, group: i_dist.Group):
    group = group.ranks_to_torch_group()
    return dist.send(tensor=x, dst=dst, group=group, tag=tag)


def recv(x_buffer: torch.Tensor, src: int, tag: int, group: i_dist.Group):
    group = group.ranks_to_torch_group()
    if x_buffer.numel() != 0:
        x_buffer = torch.empty_like(x_buffer)
    return dist.recv(x_buffer, src=src, group=group, tag=tag)
