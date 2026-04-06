"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Sequence
from models.fft_ops import build_blur_operator, fft_conv2d_circular
from models.schedule import build_difficulty_schedule
from utils.frequency import apply_lpf, compute_cts_operator_targets
# ── Base losses ─────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2):
        super().__init__()
        self.C1, self.C2, self.window_size = C1, C2, window_size
        self.register_buffer(
            "_window", self._gaussian_window(window_size, 1.5), persistent=False
        )

    @staticmethod
    def _gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target):
        C = pred.shape[1]
        w = self._window.to(pred.device, pred.dtype).expand(C, -1, -1, -1)
        pad = self.window_size // 2
        mu_x = F.conv2d(pred, w, padding=pad, groups=C)
        mu_y = F.conv2d(target, w, padding=pad, groups=C)
        s_x = F.conv2d(pred ** 2, w, padding=pad, groups=C) - mu_x ** 2
        s_y = F.conv2d(target ** 2, w, padding=pad, groups=C) - mu_y ** 2
        s_xy = F.conv2d(pred * target, w, padding=pad, groups=C) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + self.C1) * (2 * s_xy + self.C2)) / (
            (mu_x ** 2 + mu_y ** 2 + self.C1) * (s_x + s_y + self.C2)
        )
        return 1.0 - ssim.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of multiple losses."""

    def __init__(self, losses: dict[str, tuple[nn.Module, float]]):
        super().__init__()
        self.loss_modules = nn.ModuleDict()
        self.weights = {}
        for name, (mod, w) in losses.items():
            self.loss_modules[name] = mod
            self.weights[name] = w

    def forward(self, pred, target):
        total = 0.0
        for name, mod in self.loss_modules.items():
            total = total + self.weights[name] * mod(pred, target)
        return total


# ── Registry ────────────────────────────────────────────────────────

LOSS_REGISTRY: dict[str, type] = {
    "l1": nn.L1Loss,
    "l2": nn.MSELoss,
    "charbonnier": CharbonnierLoss,
    "ssim": SSIMLoss,
}


def build_loss(name: str, **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)


def build_combined_loss(specs: list[dict]) -> nn.Module:
    if len(specs) == 1:
        return build_loss(specs[0]["name"], **specs[0].get("kwargs", {}))
    losses = {}
    for s in specs:
        losses[s["name"]] = (build_loss(s["name"], **s.get("kwargs", {})), s.get("weight", 1.0))
    return CombinedLoss(losses)


# ── Stagewise loss ──────────────────────────────────────────────────

class StagewiseLoss(nn.Module):
    """Compute weighted sum of per-stage losses.

    Supports original modes: "all", "last", "one_stage", "blur_last"
    And CATS (Continuation-Aware Trajectory Supervision) modes:
      - "cats_freq":             frequency-progressive supervision
      - "cats_operator":         operator-aware closed-form targets
      - "cats_residual":         per-stage residual supervision
      - "cats_combined":         CATS-Freq primary + residual auxiliary
      - "cats_consistency":      Charbonnier + data consistency with observation y
      - "cats_consistency_all":  Charbonnier + consistency with y and all previous targets

    Weights are uniform or learnable (softmax-parameterised to stay positive and sum to 1).
    """

    def __init__(self, T: int,
                 base_loss: nn.Module,
                 learnable: bool = False,
                 mode: str = "all",
                 blur_total_sigma: float = 4.0,
                 blur_sigma_list=None,
                 # ── CATS parameters ──
                 cts_kwargs: dict | None = None,
                 kernel_size: int = -1,
                 ):
        super().__init__()
        self.T = T
        self.base_loss = base_loss
        self.learnable = learnable
        self.mode = mode
        self.kernel_size = kernel_size
        if learnable:
            self.logits = nn.Parameter(torch.zeros(T))
        else:
            self.register_buffer("logits", torch.zeros(T))

        # blur_sigmas for blur_last mode (backward compat)
        if blur_sigma_list is None or blur_sigma_list == "":
            if blur_total_sigma > 0:
                self.blur_sigmas = [
                    blur_total_sigma * math.sqrt(k / T)
                    for k in range(T - 1, -1, -1)
                ]
            else:
                self.blur_sigmas = [0.0] * T
        else:
            if isinstance(blur_sigma_list, str):
                self.blur_sigmas = [float(x) for x in blur_sigma_list.split(",") if x.strip()]
            else:
                self.blur_sigmas = [float(x) for x in blur_sigma_list]

            if len(self.blur_sigmas) != T:
                raise ValueError(
                    f"blur_sigma_list must have length T={T}, but got {len(self.blur_sigmas)}"
                )

        # ── CATS configuration ──────────────────────────────────
        cts = cts_kwargs or {}
        self._cts_filter_type = cts.get("filter_type", "gaussian")
        self._cts_residual_weight = cts.get("residual_weight", 0.0)
        self._cts_lambda_final = cts.get("lambda_final", 0.0)

        # ── Consistency loss parameters ────────────────────────
        self._consistency_weight = cts.get("consistency_weight", 0.1)
        self._consistency_p = cts.get("consistency_p", 2.0)

        # Build difficulty schedule for CATS modes (not needed for consistency modes)
        if mode.startswith("cats_") and mode not in ("cats_consistency", "cats_consistency_all"):
            difficulty_name = cts.get("difficulty_schedule", "power")
            difficulty_kw = {
                k: v for k, v in cts.items()
                if k in ("gamma", "r")
            }
            self.difficulty_schedule = build_difficulty_schedule(
                difficulty_name, T=T, **difficulty_kw,
            )
        else:
            self.difficulty_schedule = None

    @property
    def weights(self) -> torch.Tensor:
        """(T,) weights that sum to 1."""
        return torch.softmax(self.logits, dim=0)

    def _compute_cutoffs(self, device: torch.device) -> torch.Tensor:
        """Compute per-stage frequency cutoffs from difficulty schedule."""
        d = self.difficulty_schedule().to(device)  # (T,)
        # Map difficulty d ∈ [0,1] → cutoff ∈ [cutoff_min, 1.0]
        cutoff_min = 0.05
        return cutoff_min + d * (1.0 - cutoff_min)

    def _apply_consistency_blur(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Blur x by per-sample sigma (B,) using reflect-pad + FFT convolution.

        Args:
            x:     (B, C, H, W) image tensor
            sigma: (B,) per-sample blur std
        Returns:
            (B, C, H, W) blurred image
        """
        B, C, H, W = x.shape
        sigma_max = sigma.max().item()
        if sigma_max < 1e-12:
            return x
        p = max(1, int(math.ceil(3.0 * sigma_max)))
        x_pad = F.pad(x, (p, p, p, p), mode="reflect")
        Hp, Wp = H + 2 * p, W + 2 * p
        otf = build_blur_operator(
            sigma, Hp, Wp,
            kernel_size=self.kernel_size,
            device=x.device, dtype=x.dtype,
        ).otf
        y_pad = fft_conv2d_circular(x_pad, otf)
        return y_pad[:, :, p:p+H, p:p+W]

    @staticmethod
    def _consistency_loss(pred_blurred: torch.Tensor, reference: torch.Tensor,
                          p: float) -> torch.Tensor:
        """Pixel-sum-then-power consistency loss.
        When the prediction is correct, the residual should be pure Gaussian
        noise whose spatial sum is approximately zero.
        Args:
            pred_blurred: (B, C, H, W) blurred prediction
            reference:    (B, C, H, W) observation or target to compare against
            p:            exponent (default 2)
        Returns:
            scalar loss
        """
        diff = pred_blurred - reference  # (B, C, H, W)
        pixel_sum = diff.sum(dim=(1, 2, 3))  # (B,)
        return (pixel_sum.abs() ** p).mean()

    def forward(
        self,
        stage_outputs: list[torch.Tensor],
        stage_targets: list[torch.Tensor],
        *,
        x_gt: torch.Tensor | None = None,
        blur: torch.Tensor | None = None,
        blur_sigma: torch.Tensor | None = None,
        blur_sigma_deltas: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            stage_outputs: [est_of_x_{T-1}, ..., est_of_x_0]  len=T
            stage_targets: [x_{T-1}, ..., x_0]  len=T (x_0 = clean for "all" mode)
            x_gt:          (B, C, H, W) clean GT — needed for CATS modes
            blur:          (B, C, H, W) blurry input — needed for cats_operator / consistency
            blur_sigma:    (B,) blur sigmas — needed for cats_operator
            blur_sigma_deltas: (B, T) per-stage blur stds — needed for consistency modes

        Returns:
            total_loss, info_dict with per-stage losses and weights
        """
        assert len(stage_outputs) == self.T
        if stage_targets is not None:
            assert len(stage_targets) == self.T

        w = self.weights
        device = stage_outputs[0].device
        total = torch.tensor(0.0, device=device)
        per_stage = []

        # ── Resolve clean GT for CATS modes ──
        if self.mode.startswith("cats_"):
            if x_gt is None:
                # Fallback: last stage target is clean image
                assert stage_targets is not None, \
                    "CATS modes require x_gt or stage_targets"
                x_gt = stage_targets[-1]

        # ── Precompute CATS targets if needed ──
        cats_targets = None
        cutoffs = None

        if self.mode == "cats_freq" or self.mode == "cats_combined":
            cutoffs = self._compute_cutoffs(device)

        elif self.mode == "cats_operator":
            assert blur is not None and blur_sigma is not None, \
                "cats_operator mode requires blur and blur_sigma"
            d = self.difficulty_schedule().to(device)  # (T,) increase
            cats_targets = compute_cts_operator_targets(
                x_gt=x_gt, blur=blur, blur_sigma=blur_sigma, mu_schedule=1.0-d,
                kernel_size=self.kernel_size,
            )

        # ── Main loss loop ──────────────────────────────────────
        for t in range(self.T):
            if self.mode == "last":
                l_t = self.base_loss(stage_outputs[t], stage_targets[-1])

            elif self.mode == "all":
                l_t = self.base_loss(stage_outputs[t], stage_targets[t])

            elif self.mode == "one_stage":
                l_t = self.T * self.base_loss(stage_outputs[t], stage_targets[t]) \
                    if t == self.T - 1 \
                    else torch.zeros((), device=device)

            elif self.mode == "blur_last":
                loss_map = (stage_outputs[t] - stage_targets[-1]).abs()
                sigma = self.blur_sigmas[t]
                if sigma > 1e-12:
                    B, C, H, W = loss_map.shape
                    p = max(1, int(math.ceil(3.0 * sigma)))
                    loss_pad = F.pad(loss_map, (p, p, p, p), mode="reflect")
                    Hp, Wp = H + 2 * p, W + 2 * p
                    otf = build_blur_operator(
                        sigma, Hp, Wp,
                        kernel_size=self.kernel_size,
                        device=loss_map.device, dtype=loss_map.dtype,
                    ).otf
                    loss_pad = fft_conv2d_circular(loss_pad, otf)
                    loss_map = loss_pad[:, :, p:p+H, p:p+W]
                l_t = loss_map.mean()

            # ── CATS-Freq: frequency-progressive supervision ──
            elif self.mode == "cats_freq":
                cutoff = cutoffs[t].item()
                pred_lpf = apply_lpf(stage_outputs[t], cutoff, self._cts_filter_type)
                gt_lpf = apply_lpf(x_gt, cutoff, self._cts_filter_type)
                l_t = self.base_loss(pred_lpf, gt_lpf)

            # ── CATS-Operator: closed-form Fourier targets ──
            elif self.mode == "cats_operator":
                l_t = self.base_loss(stage_outputs[t], cats_targets[t])

            # ── CATS-Residual: per-stage residual supervision ──
            elif self.mode == "cats_residual":
                assert stage_targets is not None
                if t == 0:
                    # First stage residual: output - blurry input approximation
                    # Use stage_targets for the target residual
                    l_t = self.base_loss(stage_outputs[t], stage_targets[t])
                else:
                    delta_pred = stage_outputs[t] - stage_outputs[t - 1]
                    delta_target = stage_targets[t] - stage_targets[t - 1]
                    l_t = self.base_loss(delta_pred, delta_target)

            # ── CATS-Combined: Freq primary + Residual auxiliary ──
            elif self.mode == "cats_combined":
                cutoff = cutoffs[t].item()
                pred_lpf = apply_lpf(stage_outputs[t], cutoff, self._cts_filter_type)
                gt_lpf = apply_lpf(x_gt, cutoff, self._cts_filter_type)
                l_primary = self.base_loss(pred_lpf, gt_lpf)

                if t == 0 or self._cts_residual_weight <= 0:
                    l_t = l_primary
                else:
                    delta_pred = stage_outputs[t] - stage_outputs[t - 1]
                    delta_gt = apply_lpf(x_gt, cutoff, self._cts_filter_type) \
                        - apply_lpf(x_gt, cutoffs[t-1].item(), self._cts_filter_type)
                    l_residual = self.base_loss(delta_pred, delta_gt)
                    l_t = l_primary + self._cts_residual_weight * l_residual

            # ── CATS-Consistency: Charbonnier + data consistency with y ──
            elif self.mode == "cats_consistency":
                assert blur_sigma_deltas is not None, \
                    "cats_consistency mode requires blur and blur_sigma_deltas"
                l_t = self.base_loss(stage_outputs[t], stage_targets[t])
                # Residual sigma to blur output_t back to y's level:
                # deltas[:, T-1-t : T] are the (t+1) deltas already removed
                residual_sq = blur_sigma_deltas[:, self.T-1-t:self.T].pow(2).sum(dim=1)
                residual_sigma = residual_sq.sqrt()  # (B,)
                blurred = self._apply_consistency_blur(stage_outputs[t], residual_sigma)
                targeted = self._apply_consistency_blur(stage_targets[-1], blur_sigma)
                l_t = l_t + self._consistency_weight * self.base_loss(
                    blurred, targeted
                )

            # ── CATS-Consistency-All: Charbonnier + consistency with y and all previous targets ──
            elif self.mode == "cats_consistency_all":
                assert blur is not None and blur_sigma_deltas is not None, \
                    "cats_consistency_all mode requires blur and blur_sigma_deltas"
                assert stage_targets is not None
                l_t = self.base_loss(stage_outputs[t], stage_targets[t])
                w_consist = self._consistency_weight / (t + 1)

                # Compare with y
                residual_sq_y = blur_sigma_deltas[:, self.T-1-t:self.T].pow(2).sum(dim=1)
                blurred_y = self._apply_consistency_blur(stage_outputs[t], residual_sq_y.sqrt())
                targeted = self._apply_consistency_blur(stage_targets[-1], blur_sigma)
                l_t = l_t + w_consist * self.base_loss(
                    blurred_y, targeted
                )

                # Compare with each previous stage target s (0..t-1)
                for s in range(t):
                    # Deltas from T-1-t to T-2-s (t-s deltas)
                    residual_sq_s = blur_sigma_deltas[:, self.T-1-t:self.T-1-s].pow(2).sum(dim=1)
                    blurred_s = self._apply_consistency_blur(stage_outputs[t], residual_sq_s.sqrt())
                    l_t = l_t + w_consist * self.base_loss(
                        blurred_s, stage_targets[s]
                    )

            else:
                raise ValueError(f"Unknown StagewiseLoss mode: {self.mode}")

            total = total + w[t] * l_t
            per_stage.append(l_t.item())

        # ── Optional final-stage clean GT loss for CATS modes ──
        if self._cts_lambda_final > 0 and self.mode.startswith("cats_"):
            l_final = self.base_loss(stage_outputs[-1], x_gt)
            total = total + self._cts_lambda_final * l_final

        info = {
            "per_stage_loss": per_stage,
            "weights": w.detach().cpu().tolist(),
        }
        return total, info