"""FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) solver stage.

Accelerated proximal gradient with Nesterov momentum:
    y = x + ((t_k - 1) / t_{k+1}) * (x - x_prev)
    z = y - eta * grad(data_fidelity at y)
    x_new = denoiser(z)

Convergence rate: O(1/k^2) vs O(1/k) for ISTA.
"""

import torch
from .base import BaseSolver
from ..denoisers import apply_denoiser


class FISTASolver(BaseSolver):
    def step(self, x_t, denoiser, otf, beta, inner_iters=1, noise_sigma=None):
        # Step size eta = 1/beta
        eta = 1.0 / beta if not isinstance(beta, torch.Tensor) else 1.0 / beta
        if isinstance(eta, torch.Tensor) and eta.dim() >= 1:
            while eta.dim() < 4:
                eta = eta.unsqueeze(-1)

        otf_c = otf.unsqueeze(1) if otf.dim() == 3 else otf
        otf_conj = otf_c.conj()

        u = x_t
        u_prev = x_t
        t_k = 1.0

        for k in range(inner_iters):
            # Nesterov momentum
            t_k1 = (1.0 + (1.0 + 4.0 * t_k * t_k) ** 0.5) / 2.0
            momentum = (t_k - 1.0) / t_k1
            y = u + momentum * (u - u_prev)
            t_k = t_k1

            # Gradient of data fidelity at y
            Gy = torch.fft.irfft2(torch.fft.rfft2(y) * otf_c, s=y.shape[-2:])
            residual = Gy - x_t
            grad = torch.fft.irfft2(torch.fft.rfft2(residual) * otf_conj, s=y.shape[-2:])

            # Gradient descent + proximal step
            z = y - eta * grad
            u_prev = u
            u = apply_denoiser(denoiser, z, noise_sigma)

        return u
