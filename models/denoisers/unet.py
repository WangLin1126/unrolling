"""Lightweight U-Net denoiser for proximal step."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, mid_channels: int = 32, num_levels: int = 2,
                 kernel_size: int = 3, **_kwargs):
        super().__init__()
        self.num_levels = num_levels

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(num_levels):
            out_ch = mid_channels * (2 ** i)
            self.encoders.append(_DoubleConv(ch, out_ch, kernel_size=kernel_size))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        self.bottleneck = _DoubleConv(ch, ch * 2, kernel_size=kernel_size)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch * 2
        for i in range(num_levels - 1, -1, -1):
            out_ch = mid_channels * (2 ** i)
            self.upconvs.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2))
            self.decoders.append(_DoubleConv(out_ch * 2, out_ch, kernel_size=kernel_size))
            ch = out_ch

        self.head = nn.Conv2d(ch, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = 2 ** self.num_levels
        _, _, H, W = x.shape
        pH = (factor - H % factor) % factor
        pW = (factor - W % factor) % factor
        xp = F.pad(x, (0, pW, 0, pH), mode="reflect")

        skips = []
        h = xp
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h)

        for up, dec, s in zip(self.upconvs, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape != s.shape:
                h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = dec(torch.cat([h, s], dim=1))

        h = self.head(h)
        h = h[:, :, :H, :W]
        return x + h