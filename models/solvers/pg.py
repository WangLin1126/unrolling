"""Proximal Gradient (ISTA-style) solver stage."""

import torch
import torch.nn as nn
from .base import BaseSolver
from ..denoisers import apply_denoiser

class PGSolver(BaseSolver):
    def step(self, x_t, denoiser, otf, beta, inner_iters=1, noise_sigma = None):
        eta = beta
        # Reshape eta for broadcasting with (B, C, H, W) if batched
        if isinstance(eta, torch.Tensor) and eta.dim() >= 1:
            while eta.dim() < 4:
                eta = eta.unsqueeze(-1)
        otf_c = otf.unsqueeze(1) if otf.dim() == 3 else otf
        otf_conj = otf_c.conj()
        u = x_t
        for _ in range(inner_iters):
            Gu = torch.fft.irfft2(torch.fft.rfft2(u) * otf_c, s=u.shape[-2:])
            residual = Gu - x_t
            grad = torch.fft.irfft2(torch.fft.rfft2(residual) * otf_conj, s=u.shape[-2:])
            z = u - eta * grad
            u = apply_denoiser(denoiser, z, noise_sigma)
        return u