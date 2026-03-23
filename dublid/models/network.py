"""DUBLID: Deep Unrolling for Blind Image Deblurring — Motion Blur.

Faithful reproduction of the algorithm from:
  "Efficient and Interpretable Deep Blind Image Deblurring Via Algorithm Unrolling"

Architecture:
  1. Cascaded filtering: num_layers conv filters extract filtered observations
  2. Unrolled iterations (reverse order): feature update → thresholding → kernel update
  3. Final reconstruction: frequency-domain closed-form (grayscale scalar / RGB 3×3)

Reference bug fixes:
  - torch.rfft/irfft → torch.fft.rfft2/irfft2 with native complex tensors
  - fft_size reassignment removed (line 170 in ref) — use consistent padded size
  - Device parameter removed from __init__ — use .to(device) instead
  - F.upsample → F.interpolate

All learnable parameters correspond to optimization variables from the paper:
  - weight_list:      cascaded conv filter weights
  - bias_list:        soft-thresholding biases (≥ 0)
  - prox_list:        proximity / regularization strengths (≥ 0)
  - kernel_prox_list: kernel update step sizes (≥ 0)
  - kernel_bias_list: kernel normalization biases (≥ 0)
"""

from __future__ import annotations

from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops


class DUBLIDNet(nn.Module):
    """DUBLID blind deblurring network for motion blur.

    Args:
        in_channels:       input image channels (1=grayscale, 3=RGB)
        C:                 number of feature maps
        K:                 conv filter spatial size
        num_layers:        number of unrolled layers
        kernel_size:       (Hk, Wk) bounding box for blur kernel estimation
        bias_init:         initial value for thresholding biases
        zeta_init:         initial proximity param for feature update
        eta_init:          initial proximity param for reconstruction
        prox_scale:        scaling factor applied to proximity parameters
        kernel_prox_init:  initial kernel update step size
        kernel_bias_init:  initial kernel normalization bias
        kernel_scale:      temperature for logsumexp kernel normalization
        kernel_bias_scale: scaling for kernel bias
        epsilon:           small constant to prevent division by zero
    """

    def __init__(
        self,
        in_channels: int = 1,
        C: int = 16,
        K: int = 3,
        num_layers: int = 10,
        kernel_size: tuple[int, int] = (45, 45),
        bias_init: float = 0.02,
        zeta_init: float = 1.0,
        eta_init: float = 1.0,
        prox_scale: float = 10.0,
        kernel_prox_init: float = 1.0,
        kernel_bias_init: float = 0.0,
        kernel_scale: float = 1e2,
        kernel_bias_scale: float = 0.01,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.C = C
        self.K = K
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.prox_scale = prox_scale
        self.kernel_scale = kernel_scale
        self.kernel_bias_scale = kernel_bias_scale
        self.epsilon = epsilon

        # ── Cascaded conv filter weights ─────────────────────────
        self.weight_list = nn.ParameterList()
        # Layer 0: C_in → C
        w0 = nn.Parameter(nn.init.xavier_normal_(
            torch.empty(C, in_channels, K, K)))
        self.weight_list.append(w0)
        # Layers 1..(num_layers-1): C → C
        for _ in range(num_layers - 1):
            w = nn.Parameter(nn.init.xavier_normal_(
                torch.empty(C, C, 3, 3)))
            self.weight_list.append(w)

        # ── Thresholding biases (num_layers + 2 entries, all ≥ 0) ──
        self.bias_list = nn.ParameterList()
        for _ in range(num_layers + 2):
            b = nn.Parameter(torch.full((1, C, 1, 1), bias_init))
            self.bias_list.append(b)

        # ── Kernel normalization biases (num_layers entries, ≥ 0) ──
        self.kernel_bias_list = nn.ParameterList()
        for _ in range(num_layers):
            kb = nn.Parameter(torch.full((1,), kernel_bias_init))
            self.kernel_bias_list.append(kb)

        # ── Kernel update step sizes (num_layers entries, ≥ 0) ──
        self.kernel_prox_list = nn.ParameterList()
        for _ in range(num_layers):
            kp = nn.Parameter(torch.full((1,), kernel_prox_init))
            self.kernel_prox_list.append(kp)

        # ── Proximity / regularization params ────────────────────
        # prox_list[0..num_layers-1] = zeta (feature update)
        # prox_list[num_layers]      = eta  (reconstruction)
        self.prox_list = nn.ParameterList()
        for _ in range(num_layers):
            zeta = nn.Parameter(torch.full((1, C, 1, 1), zeta_init))
            self.prox_list.append(zeta)
        eta = nn.Parameter(torch.full((1, C, 1, 1), eta_init))
        self.prox_list.append(eta)

    def project_params(self) -> None:
        """Project constrained parameters to non-negative orthant.

        Call this after each optimizer step (with torch.no_grad()).
        """
        with torch.no_grad():
            for p in self.bias_list:
                p.data.relu_()
            for p in self.kernel_bias_list:
                p.data.relu_()
            for p in self.prox_list:
                p.data.relu_()
            for p in self.kernel_prox_list:
                p.data.relu_()

    @staticmethod
    def _reflect_filter(w: torch.Tensor) -> torch.Tensor:
        """Reverse filter spatially for convolution theorem compatibility.

        Input:  (C_out, C_in, Hk, Wk)
        Output: (C_out, C_in, Hk, Wk) with spatial dims reversed
        """
        return w.flip([-2, -1])

    def _compute_image_coeffs(
        self,
        Fy: torch.Tensor,
        Fg: torch.Tensor,
        Fw: torch.Tensor,
        Fk: torch.Tensor,
    ) -> torch.Tensor:
        """Solve for sharp image in frequency domain.

        Args:
            Fy: (N, C_in, H, W//2+1) complex — blurred image
            Fg: (N, C, H, W//2+1) complex — feature maps
            Fw: (C_in, C, H, W//2+1) complex — first-layer filter weights
            Fk: (N, H, W//2+1) complex — blur kernel

        Returns:
            Fx: (N, C_in, H, W//2+1) complex — estimated sharp image
        """
        eta = self.prox_scale * self.prox_list[-1]  # (1, C, 1, 1)

        if Fw.shape[0] == 1:
            # ── Grayscale: scalar closed-form ────────────────────
            Fy_0 = Fy[:, 0]  # (N, H, W//2+1)
            # conj(Fk) * Fy + eta * sum_c(conj(Fw) * Fg)
            Fk_unsq = Fk.unsqueeze(1)  # (N, 1, H, W//2+1)
            num = Fk.conj() * Fy_0 \
                + torch.sum(eta * Fw.conj() * Fg, dim=1)
            den = csquare_real(Fk) \
                + torch.sum(eta * csquare_real_4d(Fw), dim=1)
            Fx = (num / (den + self.epsilon)).unsqueeze(1)

        elif Fw.shape[0] == 3:
            # ── RGB: 3×3 matrix system via adjugate ──────────────
            Fwr, Fwg, Fwb = Fw[0:1], Fw[1:2], Fw[2:3]  # each (1, C, H, W//2+1)
            Fyr, Fyg, Fyb = Fy[:, 0], Fy[:, 1], Fy[:, 2]  # each (N, H, W//2+1)

            # Diagonal: |Fk|² + eta * sum_c |Fw_ch|²
            Fk_sq = csquare_real(Fk)  # (N, H, W//2+1)
            Crr = Fk_sq + torch.sum(eta * csquare_real_4d(Fwr), dim=1)
            Cgg = Fk_sq + torch.sum(eta * csquare_real_4d(Fwg), dim=1)
            Cbb = Fk_sq + torch.sum(eta * csquare_real_4d(Fwb), dim=1)

            # Off-diagonal: eta * sum_c conj(Fw_i) * Fw_j
            Crg = torch.sum(eta * Fwr.conj() * Fwg, dim=1)
            Crb = torch.sum(eta * Fwr.conj() * Fwb, dim=1)
            Cgb = torch.sum(eta * Fwg.conj() * Fwb, dim=1)

            # RHS: conj(Fk) * Fy_ch + eta * sum_c conj(Fw_ch) * Fg
            Fk_conj = Fk.conj()
            Br = Fk_conj * Fyr + torch.sum(eta * Fwr.conj() * Fg, dim=1)
            Bg = Fk_conj * Fyg + torch.sum(eta * Fwg.conj() * Fg, dim=1)
            Bb = Fk_conj * Fyb + torch.sum(eta * Fwb.conj() * Fg, dim=1)

            # 3×3 inverse via adjugate (cofactor matrix)
            Crg_sq = csquare_real(Crg)
            Crb_sq = csquare_real(Crb)
            Cgb_sq = csquare_real(Cgb)

            Irr = Cgg * Cbb - Cgb_sq
            Igg = Crr * Cbb - Crb_sq
            Ibb = Crr * Cgg - Crg_sq
            Irg = Cgb.conj() * Crb - Cbb * Crg
            Irb = Crg * Cgb - Cgg * Crb
            Igb = Crg.conj() * Crb - Crr * Cgb

            # Determinant
            den = (Crr * (Cgg * Cbb - Cgb_sq)
                   - Cgg * Crb_sq - Cbb * Crg_sq
                   + 2 * (Crg * Cgb * Crb.conj()).real)
            den = den + self.epsilon

            # Solve
            Fxr = Irr * Br + Irg * Bg + Irb * Bb
            Fxg = Irg.conj() * Br + Igg * Bg + Igb * Bb
            Fxb = Irb.conj() * Br + Igb.conj() * Bg + Ibb * Bb

            Fx = torch.stack([Fxr, Fxg, Fxb], dim=1) / den.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported C_in={Fw.shape[0]}, expected 1 or 3")

        return Fx

    def forward(
        self, blurred_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DUBLID forward pass.

        Args:
            blurred_image: (N, C_in, Hv, Wv) blurred input

        Returns:
            image_pred:  (N, C_in, Hv, Wv) estimated sharp image
            kernel_pred: (N, Hk, Wk) estimated blur kernel
        """
        Hk, Wk = self.kernel_size
        N, C_in, Hv, Wv = blurred_image.shape
        device = blurred_image.device

        # Convolution output sizes
        Hs, Ws = Hv + Hk - 1, Wv + Wk - 1  # 'same' w.r.t. kernel
        Hb, Wb = Hs + Hk - 1, Ws + Wk - 1  # 'full'
        fft_size = (int(ceil(Hb / 64.0) * 64), int(ceil(Wb / 64.0) * 64))

        # ═══════════════════════════════════════════════════════════
        # Stage 1: Cascaded filtering
        # ═══════════════════════════════════════════════════════════
        wy_list = []
        fy = blurred_image
        w0_unreflected = None  # save for reconstruction

        for layer in range(self.num_layers):
            w = self.weight_list[layer]
            if layer == 0:
                # Mean-subtract first-layer weights
                w_mean = w.reshape(self.C, C_in, -1).mean(dim=-1)
                w = w - w_mean.reshape(self.C, C_in, 1, 1)
                # Save unreflected weights (C_in, C, K, K) for reconstruction
                w0_unreflected = w.transpose(0, 1)
                # Reflect filter for convolution (not correlation)
                w = self._reflect_filter(w)

            fy = ops.conv2(fy, w)
            # Pad to fft_size for frequency-domain operations
            fy_padded = ops.pad_to(
                F.pad(fy, (Wk - 1, Wk - 1, Hk - 1, Hk - 1)),
                size=fft_size,
            )
            wy_list.append(fy_padded)

        # ═══════════════════════════════════════════════════════════
        # Stage 2: Unrolled kernel estimation (reverse layer order)
        # ═══════════════════════════════════════════════════════════
        # Initialize kernel as delta
        delta = torch.zeros(N, Hk, Wk, device=device)
        delta[:, Hk // 2, Wk // 2] = 1.0

        # Initialize surrogate: threshold the deepest filtered feature
        b0 = self.bias_list[0]
        z = ops.threshold(
            ops.circ_shift(wy_list[-1], (Hk // 2, Wk // 2)), b0
        )
        Fz = ops.fft2(z, size=fft_size)

        k = delta
        Fg = None  # will be set in loop

        for layer in range(self.num_layers):
            # Pop from stack (reverse order of cascaded filtering)
            fy_padded = wy_list.pop()
            Ffy = ops.fft2(fy_padded, size=fft_size)
            Fk = ops.fft2(k, size=fft_size).unsqueeze(1)  # (N, 1, H, W//2+1)

            # ── Feature update (frequency-domain closed form) ────
            zeta = self.prox_scale * self.prox_list[layer]  # (1, C, 1, 1)
            # Fg = (zeta * conj(Fk) * Ffy + Fz) / (zeta * |Fk|² + 1)
            num = zeta * ops.conj_mul(Fk, Ffy) + Fz
            den = zeta * ops.csquare(Fk) + 1.0
            Fg = num / (den + self.epsilon)

            # ── Surrogate / threshold update ─────────────────────
            b = self.bias_list[layer + 1]
            z = ops.threshold(ops.ifft2(Fg, size=fft_size), b)
            Fz = ops.fft2(z, size=fft_size)

            # ── Kernel update ────────────────────────────────────
            zk = self.kernel_prox_list[layer]
            # sum over C channels for kernel update
            num_k = zk * torch.sum(ops.conj_mul(Fz, Ffy), dim=1) \
                + Fk.squeeze(1)
            den_k = zk * torch.sum(ops.csquare(Fz), dim=1) + 1.0
            k = ops.ifft2(num_k / (den_k + self.epsilon), size=fft_size)

            # Kernel normalization with logsumexp
            k_flat = k.reshape(N, -1)
            k_max = torch.logsumexp(k_flat * self.kernel_scale, dim=-1)
            k_max = k_max / self.kernel_scale  # (N,)

            # Crop to kernel bounding box, apply ReLU and normalize
            bk = self.kernel_bias_scale * self.kernel_bias_list[layer]
            k = F.relu(k[:, :Hk, :Wk] - bk * k_max.reshape(N, 1, 1))
            k_sum = k.sum(dim=(1, 2), keepdim=True)
            k = (k + self.epsilon * delta) / (k_sum + self.epsilon)

        # ═══════════════════════════════════════════════════════════
        # Stage 3: Final image reconstruction
        # ═══════════════════════════════════════════════════════════
        Fy = ops.fft2(
            F.pad(blurred_image, (Wk - 1, Wk - 1, Hk - 1, Hk - 1)),
            size=fft_size,
        )
        Fk = ops.fft2(k, size=fft_size)  # (N, H, W//2+1)
        Fw0 = ops.fft2(
            ops.circ_shift(ops.pad_to(w0_unreflected, size=fft_size),
                           (self.K // 2, self.K // 2)),
            size=fft_size,
        )

        # Final thresholding on features
        b_final = self.bias_list[self.num_layers + 1]
        Fg = ops.fft2(ops.threshold(ops.ifft2(Fg, size=fft_size), b_final),
                       size=fft_size)

        Fx = self._compute_image_coeffs(Fy, Fg, Fw0, Fk)
        image = ops.ifft2(Fx, size=fft_size)
        # Crop to valid region
        image = image[:, :, Hk // 2:Hk // 2 + Hv, Wk // 2:Wk // 2 + Wv]

        return image, k


# ── Helpers for csquare that returns real ────────────────────────────

def csquare_real(c: torch.Tensor) -> torch.Tensor:
    """Return |c|² as a real tensor."""
    return (c * c.conj()).real


def csquare_real_4d(c: torch.Tensor) -> torch.Tensor:
    """Return |c|² as a real tensor, preserving all dims."""
    return (c * c.conj()).real
