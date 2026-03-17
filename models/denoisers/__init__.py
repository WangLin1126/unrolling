"""Denoiser (proximal network) registry."""

from __future__ import annotations
import torch.nn as nn

from .dncnn import DnCNN
from .unet import UNet
from .resblock import ResBlockDenoiser
from .drunet import DRUNet
from .uformer import UFormer
from .restormer import Restormer

DENOISER_REGISTRY: dict[str, type] = {
    "dncnn": DnCNN,
    "unet": UNet,
    "resblock": ResBlockDenoiser,
    "drunet": DRUNet,
    "uformer": UFormer,
    "restormer": Restormer,
}


def build_denoiser(name: str, **kwargs) -> nn.Module:
    if name not in DENOISER_REGISTRY:
        raise ValueError(
            f"Unknown denoiser '{name}'. Choose from {list(DENOISER_REGISTRY)}"
        )
    return DENOISER_REGISTRY[name](**kwargs)

def apply_denoiser(denoiser: nn.Module, x, noise_sigma = None):
    """Call denoiser with sigma_t when the denoiser expects it."""
    if getattr(denoiser, "requires_noise_sigma", False) and (noise_sigma is not None):
        return denoiser(x, noise_sigma)
    return denoiser(x)