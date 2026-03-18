import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return identity + out


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, kernel_size: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(in_channels, kernel_size=kernel_size) for _ in range(num_blocks)])
        self.down = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.blocks(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_blocks: int,
                 kernel_size: int = 3):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )
        pad = kernel_size // 2
        self.fuse = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.blocks = nn.Sequential(*[ResBlock(out_channels, kernel_size=kernel_size) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)
        return x


class DRUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 64,
        num_blocks: int = 4,
        residual: bool = False,
        num_levels: int = 4,
        ch_mult: float = 2.0,
        chs: Sequence[int] | None = None,
        kernel_size: int = 3,
        **_kwargs,
    ):
        """
        Args:
            in_channels: input/output image channels
            mid_channels: base channel width for level 0
            num_blocks: number of ResBlocks in each stage
            residual: if True, return x - out
            num_levels: number of channel levels, e.g. 4 -> [C, 2C, 4C, 8C]
            ch_mult: channel multiplication factor between levels
            chs: optional explicit channel list; if given, overrides num_levels/ch_mult
            kernel_size: convolution kernel size

        Examples:
            DRUNet(mid_channels=32, num_levels=4, ch_mult=2)
                -> chs = [32, 64, 128, 256]

            DRUNet(chs=[32, 48, 96, 192])
                -> use exactly these channels
        """
        super().__init__()
        self.requires_noise_sigma = True
        self.in_channels = in_channels
        self.residual = residual
        pad = kernel_size // 2

        if chs is not None:
            if len(chs) < 1:
                raise ValueError("chs must contain at least one channel value")
            self.chs = [int(c) for c in chs]
        else:
            if num_levels < 1:
                raise ValueError("num_levels must be >= 1")
            if ch_mult <= 0:
                raise ValueError("ch_mult must be > 0")
            self.chs = [int(round(mid_channels * (ch_mult ** i))) for i in range(num_levels)]

        # 输入: image + sigma_map
        self.head = nn.Conv2d(in_channels + 1, self.chs[0], kernel_size, padding=pad, bias=False)

        # encoder
        self.encoders = nn.ModuleList()
        for i in range(len(self.chs) - 1):
            self.encoders.append(DownBlock(self.chs[i], self.chs[i + 1], num_blocks, kernel_size=kernel_size))

        # bottleneck
        self.bottleneck = nn.Sequential(
            *[ResBlock(self.chs[-1], kernel_size=kernel_size) for _ in range(num_blocks)]
        )

        # decoder
        self.decoders = nn.ModuleList()
        for i in range(len(self.chs) - 1, 0, -1):
            self.decoders.append(
                UpBlock(
                    in_channels=self.chs[i],
                    skip_channels=self.chs[i - 1],
                    out_channels=self.chs[i - 1],
                    num_blocks=num_blocks,
                    kernel_size=kernel_size,
                )
            )

        self.tail = nn.Conv2d(self.chs[0], in_channels, kernel_size, padding=pad, bias=False)

    def _sigma_to_map(self, x: torch.Tensor, sigma) -> torch.Tensor:
        B, _, H, W = x.shape
        device, dtype = x.device, x.dtype

        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=device, dtype=dtype)
        else:
            sigma = sigma.to(device=device, dtype=dtype)

        if sigma.ndim == 0:
            sigma = sigma.view(1, 1, 1, 1).expand(B, 1, H, W)
        elif sigma.ndim == 1:
            if sigma.shape[0] != B:
                raise ValueError(f"sigma shape {sigma.shape} incompatible with batch size {B}")
            sigma = sigma.view(B, 1, 1, 1).expand(B, 1, H, W)
        elif sigma.ndim == 2:
            if sigma.shape != (B, 1):
                raise ValueError(f"sigma shape {sigma.shape} incompatible with batch size {B}")
            sigma = sigma.view(B, 1, 1, 1).expand(B, 1, H, W)
        elif sigma.ndim == 4:
            if sigma.shape[0] != B or sigma.shape[1] != 1:
                raise ValueError(f"sigma shape {sigma.shape} incompatible with input shape {x.shape}")
            if sigma.shape[-2:] == (1, 1):
                sigma = sigma.expand(B, 1, H, W)
            elif sigma.shape[-2:] != (H, W):
                raise ValueError(f"sigma spatial size {sigma.shape[-2:]} must be (1,1) or {(H, W)}")
        else:
            raise ValueError(f"Unsupported sigma shape: {sigma.shape}")

        return sigma

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x_in = x
        sigma_map = self._sigma_to_map(x, sigma)
        x = torch.cat([x, sigma_map], dim=1)

        x = self.head(x)

        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        out = self.tail(x)

        if self.residual:
            return x_in - out
        return out