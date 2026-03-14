"""ADMM solver stage."""

import torch
import torch.nn as nn
from .base import BaseSolver
from ..fft_ops import fft_data_step
from ..denoisers import apply_denoiser

class ADMMSolver(BaseSolver):
    def step(self, x_t, denoiser, otf, beta, inner_iters=1, noise_sigma = None):
        v = x_t
        d = torch.zeros_like(x_t)
        for _ in range(inner_iters):
            u = fft_data_step(x_t, v - d, otf, beta)
            v = apply_denoiser(denoiser, u + d, noise_sigma)
            d = d + u - v
        return v