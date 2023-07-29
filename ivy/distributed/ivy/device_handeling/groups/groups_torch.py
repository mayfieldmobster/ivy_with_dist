class TorchGroupMixin:
    def ranks_to_torch_group(self):
        import torch.distributed as t_dist

        return t_dist.new_group(self.ranks)
