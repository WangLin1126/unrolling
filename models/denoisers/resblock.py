"""Simple ResBlock-based denoiser."""

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, channels, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


class ResBlockDenoiser(nn.Module):
    def __init__(self, in_channels: int = 3, mid_channels: int = 64, num_blocks: int = 5,
                 kernel_size: int = 3, **_kwargs):
        super().__init__()
        pad = kernel_size // 2
        layers = [nn.Conv2d(in_channels, mid_channels, kernel_size, padding=pad)]
        for _ in range(num_blocks):
            layers.append(_ResBlock(mid_channels, kernel_size=kernel_size))
        layers.append(nn.Conv2d(mid_channels, in_channels, kernel_size, padding=pad))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)