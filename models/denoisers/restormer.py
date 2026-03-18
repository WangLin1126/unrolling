"""Restormer-style denoiser: Transformer with transposed (channel) attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransposedAttention(nn.Module):
    """Multi-head transposed (channel) attention from Restormer.

    Instead of spatial attention (N x N), computes channel attention (C x C),
    which is more efficient for high-resolution images.
    """

    def __init__(self, dim: int, num_heads: int = 4, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, -1, H * W)
        k = k.reshape(B, self.num_heads, -1, H * W)
        v = v.reshape(B, self.num_heads, -1, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Channel attention: (C/h, N) @ (N, C/h) -> (C/h, C/h)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, C, H, W)
        return self.project_out(out)


class GatedDconvFFN(nn.Module):
    """Gated depth-wise convolution feed-forward network."""

    def __init__(self, dim: int, ffn_expansion: float = 2.66, bias: bool = True,
                 kernel_size: int = 3):
        super().__init__()
        hidden = int(dim * ffn_expansion)
        pad = kernel_size // 2
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size, padding=pad,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * F.gelu(x2)
        return self.project_out(x)


class TransformerBlock(nn.Module):
    """Restormer Transformer block: LayerNorm + TransposedAttn + LayerNorm + GatedFFN."""

    def __init__(self, dim: int, num_heads: int = 4, ffn_expansion: float = 2.66,
                 bias: bool = True, kernel_size: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TransposedAttention(dim, num_heads=num_heads, bias=bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GatedDconvFFN(dim, ffn_expansion=ffn_expansion, bias=bias,
                                 kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # LayerNorm on channel dimension
        x = x + self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = x + self.ffn(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding for downsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample(nn.Module):
    """Pixel-unshuffle based downsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """Pixel-shuffle based upsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Restormer(nn.Module):
    """Restormer denoiser: multi-scale Transformer with transposed attention.

    Args:
        in_channels: input/output image channels
        mid_channels: base embedding dimension
        num_levels: number of encoder/decoder levels
        num_blocks: list of block counts per level, or single int
        num_heads: list of head counts per level, or single int
        ffn_expansion: FFN expansion ratio
        kernel_size: convolution kernel size for FFN and embedding
        bias: use bias in convolutions
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 48,
        num_levels: int = 3,
        num_blocks: int | list = 2,
        num_heads: int | list = 2,
        ffn_expansion: float = 2.66,
        kernel_size: int = 3,
        bias: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.num_levels = num_levels

        # Normalize to lists
        if isinstance(num_blocks, int):
            # blocks per level: encoder levels + bottleneck
            blocks_list = [num_blocks] * (num_levels + 1)
        else:
            blocks_list = list(num_blocks)
        if isinstance(num_heads, int):
            heads_list = [num_heads * (2 ** i) for i in range(num_levels + 1)]
        else:
            heads_list = list(num_heads)

        # Input projection
        self.embed = OverlapPatchEmbed(in_channels, mid_channels, kernel_size=kernel_size)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = mid_channels
        for i in range(num_levels):
            self.encoders.append(nn.Sequential(*[
                TransformerBlock(ch, num_heads=heads_list[i],
                                 ffn_expansion=ffn_expansion, bias=bias,
                                 kernel_size=kernel_size)
                for _ in range(blocks_list[i])
            ]))
            self.downsamples.append(Downsample(ch))
            ch = ch * 2

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            TransformerBlock(ch, num_heads=heads_list[num_levels],
                             ffn_expansion=ffn_expansion, bias=bias,
                             kernel_size=kernel_size)
            for _ in range(blocks_list[num_levels])
        ])

        # Decoder
        self.upsamples = nn.ModuleList()
        self.reduce_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            out_ch = ch // 2
            self.upsamples.append(Upsample(ch))
            self.reduce_convs.append(nn.Conv2d(out_ch * 2, out_ch, 1, bias=bias))
            dec_idx = i  # mirror encoder level
            self.decoders.append(nn.Sequential(*[
                TransformerBlock(out_ch, num_heads=heads_list[dec_idx],
                                 ffn_expansion=ffn_expansion, bias=bias,
                                 kernel_size=kernel_size)
                for _ in range(blocks_list[dec_idx])
            ]))
            ch = out_ch

        # Output projection
        self.output = nn.Conv2d(mid_channels, in_channels, kernel_size,
                                padding=kernel_size // 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.embed(x)

        skips = []
        for enc, down in zip(self.encoders, self.downsamples):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, reduce, dec, skip in zip(self.upsamples, self.reduce_convs,
                                         self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = reduce(torch.cat([x, skip], dim=1))
            x = dec(x)

        out = self.output(x)
        return x_in + out
