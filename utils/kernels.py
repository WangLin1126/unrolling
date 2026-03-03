"""Kernel / PSF / OTF utilities."""

import math
import torch
import torch.nn.functional as F


def gaussian_kernel2d(ks: int, sigma: float, device=None, dtype=torch.float32):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    k = k / k.sum()
    return k


def motion_kernel2d(ks: int, angle_deg: float, length: float,
                    device=None, dtype=torch.float32):
    k = torch.zeros((ks, ks), device=device, dtype=dtype)
    cx = cy = (ks - 1) / 2
    ang = math.radians(angle_deg)
    dx, dy = math.cos(ang), math.sin(ang)
    n = int(max(ks, length * 2))
    for i in range(n):
        t = (i / (n - 1) - 0.5) * length
        x = cx + t * dx
        y = cy + t * dy
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < ks and 0 <= yi < ks:
            k[yi, xi] += 1.0
    if k.sum() > 0:
        k = k / k.sum()
    else:
        k[ks // 2, ks // 2] = 1.0
    return k


def psf2otf(psf: torch.Tensor, out_h: int, out_w: int):
    """psf: (kh, kw) or (B, kh, kw) -> OTF complex (..., H, W//2+1)."""
    if psf.dim() == 2:
        psf = psf.unsqueeze(0)
    B, kh, kw = psf.shape
    pad = torch.zeros((B, out_h, out_w), device=psf.device, dtype=psf.dtype)
    pad[:, :kh, :kw] = psf
    pad = torch.roll(pad, shifts=(-(kh // 2), -(kw // 2)), dims=(1, 2))
    otf = torch.fft.rfft2(pad)
    return otf