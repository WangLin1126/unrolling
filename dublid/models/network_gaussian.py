"""DUBLID Gaussian Blur variant — free-form kernel with Gaussian initialization.

Extends the motion-blur DUBLIDNet for Gaussian blur scenarios:
  - Kernel initialized as a Gaussian (parameterized by sigma_init) instead of delta
  - Smaller default kernel bounding box (Gaussian kernels are compact)
  - Otherwise identical algorithm: cascaded filtering → unrolled iterations →
    frequency-domain reconstruction

The kernel estimation remains free-form (not constrained to Gaussian family),
giving the network flexibility to handle imperfect Gaussian blur or mixed blur.
"""

from __future__ import annotations

from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops
from .network import DUBLIDNet, csquare_real, csquare_real_4d


class DUBLIDGaussianNet(DUBLIDNet):
    """DUBLID for Gaussian blur with Gaussian kernel initialization.

    Args:
        sigma_init: initial Gaussian kernel std for kernel initialization.
                    If None, falls back to delta initialization.
        All other args are inherited from DUBLIDNet.
    """

    def __init__(
        self,
        in_channels: int = 1,
        C: int = 16,
        K: int = 3,
        num_layers: int = 10,
        kernel_size: tuple[int, int] = (21, 21),
        bias_init: float = 0.02,
        zeta_init: float = 1.0,
        eta_init: float = 1.0,
        prox_scale: float = 10.0,
        kernel_prox_init: float = 1.0,
        kernel_bias_init: float = 0.0,
        kernel_scale: float = 1e2,
        kernel_bias_scale: float = 0.01,
        epsilon: float = 1e-8,
        sigma_init: float | None = 2.0,
    ):
        super().__init__(
            in_channels=in_channels,
            C=C, K=K,
            num_layers=num_layers,
            kernel_size=kernel_size,
            bias_init=bias_init,
            zeta_init=zeta_init,
            eta_init=eta_init,
            prox_scale=prox_scale,
            kernel_prox_init=kernel_prox_init,
            kernel_bias_init=kernel_bias_init,
            kernel_scale=kernel_scale,
            kernel_bias_scale=kernel_bias_scale,
            epsilon=epsilon,
        )
        self.sigma_init = sigma_init

    def _make_gaussian_kernel(
        self, N: int, device: torch.device
    ) -> torch.Tensor:
        """Create a batch of Gaussian kernels for initialization.

        Returns:
            (N, Hk, Wk) normalized Gaussian kernel
        """
        Hk, Wk = self.kernel_size
        cy, cx = Hk // 2, Wk // 2
        y = torch.arange(Hk, device=device, dtype=torch.float32) - cy
        x = torch.arange(Wk, device=device, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        sigma = self.sigma_init
        k = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        k = k / k.sum()
        return k.unsqueeze(0).expand(N, -1, -1)

    def forward(
        self, blurred_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DUBLID forward pass with Gaussian kernel initialization.

        Args:
            blurred_image: (N, C_in, Hv, Wv) blurred input

        Returns:
            image_pred:  (N, C_in, Hv, Wv) estimated sharp image
            kernel_pred: (N, Hk, Wk) estimated blur kernel
        """
        Hk, Wk = self.kernel_size
        N, C_in, Hv, Wv = blurred_image.shape
        device = blurred_image.device

        Hs, Ws = Hv + Hk - 1, Wv + Wk - 1
        Hb, Wb = Hs + Hk - 1, Ws + Wk - 1
        fft_size = (int(ceil(Hb / 64.0) * 64), int(ceil(Wb / 64.0) * 64))

        # ═══════════════════════════════════════════════════════════
        # Stage 1: Cascaded filtering (identical to motion blur)
        # ═══════════════════════════════════════════════════════════
        wy_list = []
        fy = blurred_image
        w0_unreflected = None

        for layer in range(self.num_layers):
            w = self.weight_list[layer]
            if layer == 0:
                w_mean = w.reshape(self.C, C_in, -1).mean(dim=-1)
                w = w - w_mean.reshape(self.C, C_in, 1, 1)
                w0_unreflected = w.transpose(0, 1)
                w = self._reflect_filter(w)
            fy = ops.conv2(fy, w)
            fy_padded = ops.pad_to(
                F.pad(fy, (Wk - 1, Wk - 1, Hk - 1, Hk - 1)),
                size=fft_size,
            )
            wy_list.append(fy_padded)

        # ═══════════════════════════════════════════════════════════
        # Stage 2: Unrolled iterations — Gaussian initialization
        # ═══════════════════════════════════════════════════════════
        # Initialize kernel as Gaussian (instead of delta)
        if self.sigma_init is not None:
            k = self._make_gaussian_kernel(N, device)
        else:
            k = torch.zeros(N, Hk, Wk, device=device)
            k[:, Hk // 2, Wk // 2] = 1.0

        # Delta kernel for normalization fallback
        delta = torch.zeros(N, Hk, Wk, device=device)
        delta[:, Hk // 2, Wk // 2] = 1.0

        b0 = self.bias_list[0]
        z = ops.threshold(
            ops.circ_shift(wy_list[-1], (Hk // 2, Wk // 2)), b0
        )
        Fz = ops.fft2(z, size=fft_size)

        Fg = None

        for layer in range(self.num_layers):
            fy_padded = wy_list.pop()
            Ffy = ops.fft2(fy_padded, size=fft_size)
            Fk = ops.fft2(k, size=fft_size).unsqueeze(1)

            # Feature update
            zeta = self.prox_scale * self.prox_list[layer]
            num = zeta * ops.conj_mul(Fk, Ffy) + Fz
            den = zeta * ops.csquare(Fk) + 1.0
            Fg = num / (den + self.epsilon)

            # Surrogate / threshold update
            b = self.bias_list[layer + 1]
            z = ops.threshold(ops.ifft2(Fg, size=fft_size), b)
            Fz = ops.fft2(z, size=fft_size)

            # Kernel update
            zk = self.kernel_prox_list[layer]
            num_k = zk * torch.sum(ops.conj_mul(Fz, Ffy), dim=1) \
                + Fk.squeeze(1)
            den_k = zk * torch.sum(ops.csquare(Fz), dim=1) + 1.0
            k = ops.ifft2(num_k / (den_k + self.epsilon), size=fft_size)

            # Kernel normalization
            k_flat = k.reshape(N, -1)
            k_max = torch.logsumexp(k_flat * self.kernel_scale, dim=-1)
            k_max = k_max / self.kernel_scale

            bk = self.kernel_bias_scale * self.kernel_bias_list[layer]
            k = F.relu(k[:, :Hk, :Wk] - bk * k_max.reshape(N, 1, 1))
            k_sum = k.sum(dim=(1, 2), keepdim=True)
            k = (k + self.epsilon * delta) / (k_sum + self.epsilon)

        # ═══════════════════════════════════════════════════════════
        # Stage 3: Final reconstruction (identical to motion blur)
        # ═══════════════════════════════════════════════════════════
        Fy = ops.fft2(
            F.pad(blurred_image, (Wk - 1, Wk - 1, Hk - 1, Hk - 1)),
            size=fft_size,
        )
        Fk = ops.fft2(k, size=fft_size)
        Fw0 = ops.fft2(
            ops.circ_shift(ops.pad_to(w0_unreflected, size=fft_size),
                           (self.K // 2, self.K // 2)),
            size=fft_size,
        )

        b_final = self.bias_list[self.num_layers + 1]
        Fg = ops.fft2(ops.threshold(ops.ifft2(Fg, size=fft_size), b_final),
                       size=fft_size)

        Fx = self._compute_image_coeffs(Fy, Fg, Fw0, Fk)
        image = ops.ifft2(Fx, size=fft_size)
        image = image[:, :, Hk // 2:Hk // 2 + Hv, Wk // 2:Wk // 2 + Wv]

        return image, k
