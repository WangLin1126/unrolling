"""UFormer-style denoiser: U-shaped Transformer with LeWin attention blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LeWinAttention(nn.Module):
    """Locally-enhanced Window (LeWin) self-attention."""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij")).flatten(1)  # (2, ws*ws)
        relative_coords = coords[:, :, None] - coords[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*nW, ws*ws, C)"""
        B_nW, N, C = x.shape
        qkv = self.qkv(x).reshape(B_nW, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B_nW, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_nW, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LeWinBlock(nn.Module):
    """Transformer block with LeWin attention and depth-wise conv FFN."""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8,
                 mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LeWinAttention(dim, num_heads=num_heads, window_size=window_size,
                                   attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        # Depth-wise conv for local enhancement
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)

    def _window_partition(self, x: torch.Tensor, ws: int):
        """(B, H, W, C) -> (B*nW, ws, ws, C)"""
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)

    def _window_reverse(self, windows: torch.Tensor, ws: int, H: int, W: int):
        """(B*nW, ws, ws, C) -> (B, H, W, C)"""
        B = windows.shape[0] // (H // ws * W // ws)
        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        ws = self.window_size

        # Pad to multiples of window_size
        pH = (ws - H % ws) % ws
        pW = (ws - W % ws) % ws
        if pH > 0 or pW > 0:
            x = F.pad(x, (0, pW, 0, pH), mode="reflect")
        _, _, Hp, Wp = x.shape

        # (B, C, H, W) -> (B, H, W, C)
        x_hw = x.permute(0, 2, 3, 1).contiguous()

        # Window attention
        shortcut = x_hw
        x_norm = self.norm1(x_hw)
        windows = self._window_partition(x_norm, ws)  # (B*nW, ws, ws, C)
        windows = windows.view(-1, ws * ws, C)
        attn_out = self.attn(windows)
        attn_out = attn_out.view(-1, ws, ws, C)
        x_hw = shortcut + self._window_reverse(attn_out, ws, Hp, Wp)

        # FFN with depth-wise conv
        shortcut = x_hw
        x_norm = self.norm2(x_hw)
        ffn_out = self.ffn(x_norm)
        # Local enhancement via depth-wise conv
        x_conv = x_hw.permute(0, 3, 1, 2).contiguous()
        x_conv = self.dwconv(x_conv)
        x_conv = x_conv.permute(0, 2, 3, 1).contiguous()
        x_hw = shortcut + ffn_out + x_conv

        # Back to (B, C, H, W)
        out = x_hw.permute(0, 3, 1, 2).contiguous()
        if pH > 0 or pW > 0:
            out = out[:, :, :H, :W]
        return out


class UFormer(nn.Module):
    """UFormer denoiser: U-shaped Transformer with LeWin attention.

    Args:
        in_channels: input/output image channels
        mid_channels: base embedding dimension
        num_levels: number of encoder/decoder levels
        num_blocks: number of LeWin blocks per level
        num_heads: attention heads (scales with level)
        window_size: window size for local attention
        kernel_size: convolution kernel size for down/up sampling
        mlp_ratio: FFN hidden ratio
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 32,
        num_levels: int = 2,
        num_blocks: int = 2,
        num_heads: int = 2,
        window_size: int = 8,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        **_kwargs,
    ):
        super().__init__()
        self.num_levels = num_levels
        pad = kernel_size // 2

        self.embed = nn.Conv2d(in_channels, mid_channels, kernel_size, padding=pad, bias=False)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = mid_channels
        for i in range(num_levels):
            heads = max(1, num_heads * (2 ** i))
            blocks = nn.Sequential(*[
                LeWinBlock(ch, num_heads=heads, window_size=window_size, mlp_ratio=mlp_ratio)
                for _ in range(num_blocks)
            ])
            self.encoders.append(blocks)
            next_ch = ch * 2
            self.downsamples.append(nn.Conv2d(ch, next_ch, 4, stride=2, padding=1, bias=False))
            ch = next_ch

        # Bottleneck
        bot_heads = max(1, num_heads * (2 ** num_levels))
        self.bottleneck = nn.Sequential(*[
            LeWinBlock(ch, num_heads=bot_heads, window_size=window_size, mlp_ratio=mlp_ratio)
            for _ in range(num_blocks)
        ])

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        for i in range(num_levels - 1, -1, -1):
            out_ch = mid_channels * (2 ** i) if i > 0 else mid_channels
            self.upsamples.append(nn.ConvTranspose2d(ch, out_ch, 2, stride=2, bias=False))
            self.fuse_convs.append(nn.Conv2d(out_ch * 2, out_ch, 1, bias=False))
            heads = max(1, num_heads * (2 ** i))
            blocks = nn.Sequential(*[
                LeWinBlock(out_ch, num_heads=heads, window_size=window_size, mlp_ratio=mlp_ratio)
                for _ in range(num_blocks)
            ])
            self.decoders.append(blocks)
            ch = out_ch

        self.output = nn.Conv2d(mid_channels, in_channels, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = self.embed(x)

        skips = []
        for enc, down in zip(self.encoders, self.downsamples):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for up, fuse, dec, skip in zip(self.upsamples, self.fuse_convs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = fuse(torch.cat([x, skip], dim=1))
            x = dec(x)

        out = self.output(x)
        return x_in + out
