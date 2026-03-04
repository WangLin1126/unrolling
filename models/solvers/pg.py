"""Proximal Gradient (ISTA-style) solver stage."""

import torch
import torch.nn as nn
from .base import BaseSolver


class PGSolver(BaseSolver):
    def step(self, x_t, denoiser, otf, beta, inner_iters=1):
        eta = beta
        otf_c = otf.unsqueeze(1) if otf.dim() == 3 else otf
        otf_conj = otf_c.conj()
        u = x_t
        for _ in range(inner_iters):
            Gu = torch.fft.irfft2(torch.fft.rfft2(u) * otf_c, s=u.shape[-2:])
            residual = Gu - x_t
            grad = torch.fft.irfft2(torch.fft.rfft2(residual) * otf_conj, s=u.shape[-2:])
            z = u - eta * grad
            u = denoiser(z)
        return u