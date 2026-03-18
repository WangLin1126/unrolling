"""DnCNN-style feed-forward denoiser (residual learning)."""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, in_channels: int = 3, mid_channels: int = 64, depth: int = 8,
                 kernel_size: int = 3, stride: int = 1, **_kwargs):
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size, stride=stride, padding=pad, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=pad, bias=False))
            layers.append(nn.BatchNorm2d(mid_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(mid_channels, in_channels, kernel_size, padding=pad, bias=False))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.body(x)