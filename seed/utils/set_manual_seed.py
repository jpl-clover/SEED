import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_manual_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        # cudnn.deterministic = True
        # warnings.warn(
        #     "You have chosen to seed training. "
        #     "This will turn on the CUDNN deterministic setting, "
        #     "which can slow down your training considerably! "
        #     "You may see unexpected behavior when restarting "
        #     "from checkpoints."
        # )
