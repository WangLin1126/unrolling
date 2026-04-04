"""Frequency-domain utilities for CATS (Continuation-Aware Trajectory Supervision).

Provides:
  - apply_lpf:              Low-pass filter images in frequency domain
  - radial_average_psd:     Radially averaged power spectral density
  - frequency_band_error:   Per-frequency-band MSE between prediction and target
  - compute_cts_operator_target: Closed-form CATS-Operator targets in Fourier domain
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from models.fft_ops import precompute_freq_sq, build_blur_operator


# ── Low-pass filtering ──────────────────────────────────────────────

def _build_radial_mask(
    H: int, W: int, cutoff: float,
    filter_type: str = "gaussian",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a radial frequency mask for rfft2 output shape (H, W//2+1).

    Args:
        H, W:        spatial dimensions of the *original* image
        cutoff:      cutoff as fraction of Nyquist (0 = DC only, 1 = full bandwidth)
        filter_type: "gaussian" | "butterworth" | "ideal"
        device, dtype: tensor placement

    Returns:
        (H, W//2+1) real-valued mask in [0, 1]
    """
    freq_sq = precompute_freq_sq(H, W, device=device, dtype=dtype)
    # Nyquist squared: max possible freq_sq = (π)^2 + (π)^2 = 2π²
    nyquist_sq = 2.0 * math.pi ** 2
    # Normalised squared frequency in [0, 1]
    freq_norm_sq = freq_sq / nyquist_sq

    cutoff = max(cutoff, 1e-8)
    cutoff_sq = cutoff ** 2

    if filter_type == "gaussian":
        # Gaussian roll-off: M(ω) = exp(-ω²_norm / (2·cutoff²))
        mask = torch.exp(-freq_norm_sq / (2.0 * cutoff_sq))
    elif filter_type == "butterworth":
        # Butterworth order 6 for smooth roll-off
        order = 6
        mask = 1.0 / (1.0 + (freq_norm_sq / cutoff_sq) ** order)
    elif filter_type == "ideal":
        mask = (freq_norm_sq <= cutoff_sq).to(dtype)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    return mask


def apply_lpf(
    x: torch.Tensor,
    cutoff: float,
    filter_type: str = "gaussian",
) -> torch.Tensor:
    """Apply a radial low-pass filter to images.

    Args:
        x:           (B, C, H, W) images
        cutoff:      cutoff as fraction of Nyquist (0→DC, 1→full)
        filter_type: "gaussian" | "butterworth" | "ideal"

    Returns:
        (B, C, H, W) filtered images
    """
    if cutoff >= 1.0 - 1e-6:
        return x  # no filtering needed

    _, _, H, W = x.shape
    mask = _build_radial_mask(H, W, cutoff, filter_type, x.device, x.dtype)
    # mask: (H, W//2+1) → broadcast over (B, C)
    X = torch.fft.rfft2(x)
    X_filtered = X * mask.unsqueeze(0).unsqueeze(0)
    return torch.fft.irfft2(X_filtered, s=(H, W))


# ── CATS-Operator target computation ─────────────────────────────────

def compute_cts_operator_targets(
    x_gt: torch.Tensor,
    blur: torch.Tensor,
    blur_sigma: torch.Tensor,
    mu_schedule: torch.Tensor,
    kernel_size: int = -1,
) -> list[torch.Tensor]:
    """Compute CATS-Operator closed-form targets in Fourier domain.

    Target_t = IFFT{ [μ_t |H|² Y + (1-μ_t) X_gt] / [μ_t |H|² + (1-μ_t)] }
    μ ≈ 0 → target ≈ clean X_gt (hard)
    μ ≈ 1 → target ≈ blurry observation (easy)
    Args:
        x_gt:        (B, C, H, W) clean ground truth
        blur:        (B, C, H, W) blurry observation
        blur_sigma:  (B,) per-sample total blur sigma
        mu_schedule: (T,) monotonically increasing values in [0, 1]
        kernel_size: -1 for analytic Gaussian, >0 for truncated kernel

    Returns:
        list of T tensors, each (B, C, H, W)
    """
    B, C, H, W = x_gt.shape
    T = mu_schedule.shape[0]

    X_gt = torch.fft.rfft2(x_gt)    # (B, C, H, W//2+1) complex
    Y = torch.fft.rfft2(blur)       # (B, C, H, W//2+1) complex

    # Compute |H(ω)|² via unified blur operator builder
    blur_op = build_blur_operator(
        blur_sigma, H, W,
        kernel_size=kernel_size,
        device=x_gt.device, dtype=x_gt.dtype,
    )
    H_abs_sq = (blur_op.otf * blur_op.otf.conj()).real  # (B, H, W//2+1)
    H_abs_sq = H_abs_sq.unsqueeze(1)  # (B, 1, H, W//2+1) for broadcasting with C

    targets = []
    for t in range(T):
        mu = mu_schedule[t].item()
        num = mu * H_abs_sq * Y + (1.0 - mu) * X_gt
        den = mu * H_abs_sq + (1.0 - mu) + 1e-8
        X_t = num / den
        targets.append(torch.fft.irfft2(X_t, s=(H, W)))

    return targets


# ── Analysis utilities ───────────────────────────────────────────────

def radial_average_psd(
    x: torch.Tensor, num_bins: int = 32,
) -> torch.Tensor:
    """Compute radially averaged power spectral density.

    Args:
        x:        (B, C, H, W) or (C, H, W)
        num_bins: number of radial frequency bins

    Returns:
        (num_bins,) radially averaged PSD
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    B, C, H, W = x.shape

    X = torch.fft.rfft2(x)
    psd = (X.real ** 2 + X.imag ** 2).mean(dim=(0, 1))  # (H, W//2+1)

    freq_sq = precompute_freq_sq(H, W, x.device, x.dtype)
    freq_r = torch.sqrt(freq_sq)
    max_freq = freq_r.max()

    bin_edges = torch.linspace(0, max_freq.item(), num_bins + 1, device=x.device)
    result = torch.zeros(num_bins, device=x.device, dtype=x.dtype)

    for i in range(num_bins):
        mask = (freq_r >= bin_edges[i]) & (freq_r < bin_edges[i + 1])
        if mask.any():
            result[i] = psd[mask].mean()

    return result


def frequency_band_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_bands: int = 8,
) -> torch.Tensor:
    """Compute per-frequency-band MSE between pred and target.

    Args:
        pred:   (B, C, H, W) or (C, H, W)
        target: same shape as pred
        num_bands: number of frequency bands

    Returns:
        (num_bands,) MSE per band
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    B, C, H, W = pred.shape
    E = torch.fft.rfft2(pred - target)
    error_psd = (E.real ** 2 + E.imag ** 2).mean(dim=(0, 1))  # (H, W//2+1)

    freq_sq = precompute_freq_sq(H, W, pred.device, pred.dtype)
    freq_r = torch.sqrt(freq_sq)
    max_freq = freq_r.max()

    bin_edges = torch.linspace(0, max_freq.item(), num_bands + 1, device=pred.device)
    result = torch.zeros(num_bands, device=pred.device, dtype=pred.dtype)

    for i in range(num_bands):
        mask = (freq_r >= bin_edges[i]) & (freq_r < bin_edges[i + 1])
        if mask.any():
            result[i] = error_psd[mask].mean()

    return result
