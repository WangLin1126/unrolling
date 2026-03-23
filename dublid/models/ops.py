"""Modernized FFT and complex operations for DUBLID.

Replaces the deprecated torch.rfft / torch.irfft API (removed in PyTorch 2.0)
with torch.fft.rfft2 / torch.fft.irfft2 and native complex tensors.

Reference bug fixes applied:
  - torch.rfft  → torch.fft.rfft2  (native complex output)
  - torch.irfft → torch.fft.irfft2 (native complex input)
  - F.upsample  → F.interpolate
  - All manual real/imag stack operations replaced by native complex arithmetic
"""

import torch
import torch.nn.functional as F


# ── FFT / IFFT ──────────────────────────────────────────────────────

def pad_to(original: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Post-pad last two dimensions to `size`."""
    h_pad = size[0] - original.shape[-2]
    w_pad = size[1] - original.shape[-1]
    return F.pad(original, (0, w_pad, 0, h_pad))


def fft2(signal: torch.Tensor,
         size: tuple[int, int] | None = None) -> torch.Tensor:
    """2D real FFT on the last two dimensions.

    Args:
        signal: real tensor (..., H, W)
        size: optional pad-to size before FFT

    Returns:
        complex tensor (..., H, W//2+1)
    """
    if size is not None:
        signal = pad_to(signal, size)
    return torch.fft.rfft2(signal)


def ifft2(signal: torch.Tensor,
          size: tuple[int, int] | None = None) -> torch.Tensor:
    """2D inverse real FFT on the last two dimensions.

    Args:
        signal: complex tensor (..., H, W//2+1)
        size: output spatial size (H, W)

    Returns:
        real tensor (..., H, W)
    """
    return torch.fft.irfft2(signal, s=size)


# ── Complex arithmetic (trivial with native complex) ────────────────

def conj_mul(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """conj(c1) * c2."""
    return c1.conj() * c2


def csquare(c: torch.Tensor) -> torch.Tensor:
    """|c|^2 = c * conj(c), returns real."""
    return (c * c.conj()).real


def real_mul(r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Multiply real tensor r with complex tensor c."""
    return r * c


def cmul(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """Complex multiplication c1 * c2."""
    return c1 * c2


# ── Spatial operations ──────────────────────────────────────────────

def circ_shift(ts: torch.Tensor,
               shift: tuple[int, int]) -> torch.Tensor:
    """Circular shift on the last two dimensions."""
    sr, sc = shift
    if sc != 0:
        ts = torch.cat((ts[..., sc:], ts[..., :sc]), dim=-1)
    if sr != 0:
        ts = torch.cat((ts[..., sr:, :], ts[..., :sr, :]), dim=-2)
    return ts


def conv2(tensor: torch.Tensor, kernel: torch.Tensor,
          mode: str = 'same', pad_mode: str = 'reflect') -> torch.Tensor:
    """2D convolution with configurable output size mode.

    Args:
        tensor: (N, C_in, H, W)
        kernel: (C_out, C_in, Hk, Wk)
        mode: 'same', 'full', or 'valid'
        pad_mode: padding mode for F.pad
    """
    Hk, Wk = kernel.shape[-2], kernel.shape[-1]
    if mode == 'same':
        pad_size = (Wk // 2, Wk - Wk // 2 - 1, Hk // 2, Hk - Hk // 2 - 1)
    elif mode == 'full':
        pad_size = (Wk - 1, Wk - 1, Hk - 1, Hk - 1)
    else:  # 'valid'
        pad_size = (0, 0, 0, 0)
    return F.conv2d(F.pad(tensor, pad=pad_size, mode=pad_mode), kernel)


def threshold(x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    """Soft-thresholding: sign(x) * max(|x| - thr, 0)."""
    return F.relu(x - thr) - F.relu(-x - thr)
