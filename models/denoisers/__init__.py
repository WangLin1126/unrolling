"""Denoiser (proximal network) registry."""

from __future__ import annotations
import torch.nn as nn

from .dncnn import DnCNN
from .unet_small import SmallUNet
from .resblock import ResBlockDenoiser

DENOISER_REGISTRY: dict[str, type] = {
    "dncnn": DnCNN,
    "unet": SmallUNet,
    "resblock": ResBlockDenoiser,
}


def build_denoiser(name: str, **kwargs) -> nn.Module:
    if name not in DENOISER_REGISTRY:
        raise ValueError(
            f"Unknown denoiser '{name}'. Choose from {list(DENOISER_REGISTRY)}"
        )
    return DENOISER_REGISTRY[name](**kwargs)