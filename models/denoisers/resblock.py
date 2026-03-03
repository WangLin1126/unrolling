"""Simple ResBlock-based denoiser."""

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


class ResBlockDenoiser(nn.Module):
    def __init__(self, in_channels: int = 3, mid_channels: int = 64, num_blocks: int = 5, **_kwargs):
        super().__init__()
        layers = [nn.Conv2d(in_channels, mid_channels, 3, padding=1)]
        for _ in range(num_blocks):
            layers.append(_ResBlock(mid_channels))
        layers.append(nn.Conv2d(mid_channels, in_channels, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)