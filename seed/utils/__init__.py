import torch

from .parse_args import parse_args
from .set_manual_seed import set_manual_seed
from .yaml_config_hook import yaml_config_hook
from .init_distributed_mode import init_distributed_mode

# multi-GPU data collector
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
