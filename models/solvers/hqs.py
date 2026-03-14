"""HQS (Half-Quadratic Splitting) solver stage."""

import torch
import torch.nn as nn
from .base import BaseSolver
from ..fft_ops import fft_data_step
from ..denoisers import apply_denoiser

class HQSSolver(BaseSolver):
    def step(self, x_t, denoiser, otf, beta, inner_iters = 1, noise_sigma = None):
        v = x_t
        for _ in range(inner_iters):
            u = fft_data_step(x_t, v, otf, beta)
            v = apply_denoiser(denoiser, u, noise_sigma)
        return v