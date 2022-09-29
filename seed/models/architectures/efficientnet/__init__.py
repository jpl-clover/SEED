__version__ = "0.7.0"
from .model import EfficientNet
from .utils import BlockArgs, BlockDecoder, GlobalParams, efficientnet, get_model_params

__all__ = [
    "seed_efficientnet_b0",
    "seed_efficientnet_b1",
    "seed_efficientnet_b2",
    "seed_efficientnet_b3",
]


def seed_efficientnet_b0(**kwargs):
    r"""Efficientnet model from
    `"https://github.com/lukemelas/EfficientNet-PyTorch`_
    """
    return EfficientNet.from_name("efficientnet-b0", **kwargs)


def seed_efficientnet_b1(**kwargs):
    r"""Efficientnet model from
    `"https://github.com/lukemelas/EfficientNet-PyTorch`_
    """
    return EfficientNet.from_name("efficientnet-b1", **kwargs)


def seed_efficientnet_b2(**kwargs):
    r"""Efficientnet model from
    `"https://github.com/lukemelas/EfficientNet-PyTorch`_
    """
    return EfficientNet.from_name("efficientnet-b2", **kwargs)


def seed_efficientnet_b3(**kwargs):
    r"""Efficientnet model from
    `"https://github.com/lukemelas/EfficientNet-PyTorch`_
    """
    return EfficientNet.from_name("efficientnet-b3", **kwargs)
