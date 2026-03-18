"""ISTA (Iterative Shrinkage-Thresholding Algorithm) solver stage.

Classic proximal gradient with fixed step size:
    z = x_t - eta * grad(data_fidelity)
    x_{t+1} = denoiser(z)

This is mathematically equivalent to PG but explicitly named for the
ISTA framework convention, using 1/beta as step size.
"""

import torch
from .base import BaseSolver
from ..denoisers import apply_denoiser


class ISTASolver(BaseSolver):
    def step(self, x_t, denoiser, otf, beta, inner_iters=1, noise_sigma=None):
        # Step size eta = 1/beta (inverse regularization weight)
        eta = 1.0 / beta if not isinstance(beta, torch.Tensor) else 1.0 / beta
        if isinstance(eta, torch.Tensor) and eta.dim() >= 1:
            while eta.dim() < 4:
                eta = eta.unsqueeze(-1)

        otf_c = otf.unsqueeze(1) if otf.dim() == 3 else otf
        otf_conj = otf_c.conj()

        u = x_t
        for _ in range(inner_iters):
            # Gradient of data fidelity: G^H (G u - x_t)
            Gu = torch.fft.irfft2(torch.fft.rfft2(u) * otf_c, s=u.shape[-2:])
            residual = Gu - x_t
            grad = torch.fft.irfft2(torch.fft.rfft2(residual) * otf_conj, s=u.shape[-2:])
            # Gradient descent step
            z = u - eta * grad
            # Proximal (denoiser) step
            u = apply_denoiser(denoiser, z, noise_sigma)
        return u
