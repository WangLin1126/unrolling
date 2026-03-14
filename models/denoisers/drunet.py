import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return identity + out


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock(in_channels) for _ in range(num_blocks)])
        self.down = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.blocks(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, num_blocks: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=False
        )
        self.fuse = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[ResBlock(out_channels) for _ in range(num_blocks)])

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
        **_kwargs,
    ):
        super().__init__()
        self.requires_noise_sigma = True
        self.in_channels = in_channels
        self.residual = residual

        chs = (mid_channels, mid_channels * 2, mid_channels * 4, mid_channels * 8)

        # input = image + sigma_map
        self.head = nn.Conv2d(in_channels + 1, chs[0], 3, padding=1, bias=False)

        self.enc1 = DownBlock(chs[0], chs[1], num_blocks)
        self.enc2 = DownBlock(chs[1], chs[2], num_blocks)
        self.enc3 = DownBlock(chs[2], chs[3], num_blocks)

        self.bottleneck = nn.Sequential(*[ResBlock(chs[3]) for _ in range(num_blocks)])

        self.dec3 = UpBlock(chs[3], chs[2], chs[2], num_blocks)
        self.dec2 = UpBlock(chs[2], chs[1], chs[1], num_blocks)
        self.dec1 = UpBlock(chs[1], chs[0], chs[0], num_blocks)

        self.tail = nn.Conv2d(chs[0], in_channels, 3, padding=1, bias=False)

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

        x0 = self.head(x)

        x1, s1 = self.enc1(x0)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)

        x4 = self.bottleneck(x3)

        x = self.dec3(x4, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        out = self.tail(x)

        if self.residual:
            return x_in - out
        return out