"""ADMM solver stage."""

import torch
import torch.nn as nn
from .base import BaseSolver
from ..fft_ops import fft_data_step


class ADMMSolver(BaseSolver):
    def step(self, x_t, denoiser, otf, beta, lam, inner_iters=1):
        v = x_t
        d = torch.zeros_like(x_t)
        for _ in range(inner_iters):
            u = fft_data_step(x_t, v - d, otf, beta)
            v = denoiser(u + d)
            d = d + u - v
        return v