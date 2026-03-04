"""HQS (Half-Quadratic Splitting) solver stage."""

import torch
import torch.nn as nn
from .base import BaseSolver
from ..fft_ops import fft_data_step


class HQSSolver(BaseSolver):
    def step(self, x_t, denoiser, otf, beta, inner_iters=1):
        v = x_t
        for _ in range(inner_iters):
            u = fft_data_step(x_t, v, otf, beta)
            v = denoiser(u)
        return v