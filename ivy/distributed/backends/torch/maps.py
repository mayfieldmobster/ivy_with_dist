from typing import Union, Callable, Hashable, Sequence
from functools import wraps

import torch
import torch.distributed as dist

import ivy
import ivy.distributed as i_dist


# TODO accept nestwed inputs cmap(f)((x,y),z)
def data_frag(*args, in_axes: Union[int, tuple], num_devices: int):
    new_args = [[] for _ in range(num_devices)]
    axes = (in_axes,) * len(args) if isinstance(in_axes, int) else in_axes
    if len(in_axes) != len(args):
        raise ValueError("len of in_axes must match len of args")
    for d, a in zip(axes, args):
        if not isinstance(a, torch.Tensor):
            raise TypeError("Only tensors can be mapped")
        if d is not None:
            a = ivy.split(a, num_or_size_splits=num_devices, axis=d)
            for i in range(num_devices):
                new_args[i].append(a[i])
        else:
            for i in range(num_devices):
                new_args[i].append(a)

    return new_args


def data_to_device(*args):
    return [a.to(torch.cuda.current_device()) for a in args]


def scalar_to_vec(*args):
    # if output is scalar convert to vecotr of shape (1,1) for all_gather
    return tuple(map(lambda x: x.unsqueeze(0) if x.dim() == 0 else x, args))


def pmap(
    fn: Callable,
    axis_name: Hashable,
    *,
    in_axes: Union[int, Sequence[int]] = 0,
    out_axes: Union[int, Sequence[int]] = 0,
    group: Union[i_dist.Group, None] = None,
    dst: int = 0
) -> Callable:
    if isinstance(group, i_dist.Group):
        group = group.ranks_to_torch_group()

    rank = dist.get_rank(group=group)
    num_processes = dist.get_world_size(group=group)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if rank == -1:
            # process not in group
            return None

        if num_processes == 1:
            return ivy.vmap(fn, in_axes=in_axes, out_axes=out_axes)(*args, **kwargs)

        # Run Function
        new_args = data_frag(*args, in_axes=in_axes, num_devices=num_processes)
        data_to_device(*new_args[rank])
        func_out = ivy.vmap(fn, in_axes=in_axes, out_axes=out_axes)(
            *new_args[rank], **kwargs
        )
        func_out = scalar_to_vec(*func_out)

        out_axes_ = (
            (out_axes,) * len(func_out) if isinstance(out_axes, int) else out_axes
        )

        # collect function outputs
        # TODO cleanup (lots of repetitive code)
        if dst == -1:
            if isinstance(func_out, tuple):
                output_empties = [
                    [torch.empty_like(i) for _ in range(num_processes)]
                    for i in func_out
                ]
                for i in range(len(output_empties)):
                    i_dist.all_gather(output_empties[i], func_out[i], group=group)
                    torch.cuda.synchronize()
                for i in range(len(out_axes)):
                    output_empties[i] = torch.cat(output_empties[i], dim=out_axes_[i])
                return tuple(output_empties)
            else:
                output_empties = [
                    torch.empty_like(func_out) for _ in range(num_processes)
                ]
                i_dist.all_gather(output_empties, func_out, group=group)
                torch.cuda.synchronize()
                return torch.cat(output_empties, dim=out_axes_)
        else:
            # TODO replace dist.gather with ivy.gather once supported
            if rank == dst:
                if isinstance(func_out, tuple):
                    output_empties = [
                        [torch.empty_like(i) for _ in range(num_processes)]
                        for i in func_out
                    ]
                    for i in range(len(output_empties)):
                        dist.gather(
                            func_out[i], output_empties[i], dst=dst, group=group
                        )
                        torch.cuda.synchronize()
                    for i in range(len(out_axes_)):
                        output_empties[i] = torch.cat(
                            output_empties[i], dim=out_axes_[i]
                        )
                    return tuple(output_empties)
                else:
                    output_empties = [
                        torch.empty_like(func_out) for _ in range(num_processes)
                    ]
                    dist.gather(func_out, output_empties, dst=dst, group=group)
                    torch.cuda.synchronize()
                    return torch.cat(output_empties, dim=out_axes_)
            else:
                if isinstance(func_out, tuple):
                    for i in range(len(func_out)):
                        dist.gather(func_out[i], dst=dst, group=group)
                        torch.cuda.synchronize()
                    return None
                else:
                    dist.gather(func_out, dst=dst, group=group)
                    torch.cuda.synchronize()
                    return None

    return wrapper


def xmap():
    ...
