"""Unrolled iterative deblurring with per-stage intermediate supervision.

Two modes for supervision targets:
  1. Precomputed (uniform schedule): targets come from dataset, just send to GPU
  2. On-the-fly (trainable schedule): deltas change each step, must recompute

Boundary handling (consistent with dataset):
    Dataset:  reflect-pad → FFT → crop → artifact-free y + precomputed targets
    Model:    reflect-pad(y) → deconv on (H+2p, W+2p) → crop → outputs on (H, W)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedule import build_schedule, build_beta_schedule
from .denoisers import build_denoiser
from .solvers import build_solver
from .fft_ops import gaussian_otf, fft_conv2d_circular


class UnrolledDeblurNet(nn.Module):
    """Unrolled multi-stage Gaussian deblurring with intermediate supervision.

    Args:
        T:              number of unrolling stages
        solver_name:    "hqs" | "admm" | "pg"
        schedule_name:  "uniform" | "trainable"
        denoiser_name:  "dncnn" | "unet_small" | "resblock"
        share_denoisers: share one denoiser across all stages
        inner_iters:    inner solver iterations per stage
        in_channels:    image channels
        pad_border:     reflect-pad pixels for FFT (MUST match dataset)
        denoiser_kwargs: kwargs for denoiser construction
        schedule_kwargs: kwargs for schedule construction
    """

    def __init__(
        self,
        T: int = 5,
        solver_name: str = "hqs",
        schedule_name: str = "uniform",
        denoiser_name: str = "dncnn",
        share_denoisers: bool = False,
        inner_iters: int = 1,
        in_channels: int = 3,
        pad_border: int = 32,
        denoiser_kwargs: dict | None = None,
        schedule_kwargs: dict | None = None,
        beta_kwargs: dict | None = None,
        beta_mode: str = "geom",
    ):
        super().__init__()
        self.T = T
        self.inner_iters = inner_iters
        self.pad_border = pad_border
        self.schedule_name = schedule_name

        self.delta_schedule = build_schedule(schedule_name, T=T, **(schedule_kwargs or {}))
        self.solver = build_solver(solver_name)

        dk = dict(in_channels=in_channels, **(denoiser_kwargs or {}))
        if share_denoisers:
            single = build_denoiser(denoiser_name, **dk)
            self.denoisers = nn.ModuleList([single] * T)
        else:
            self.denoisers = nn.ModuleList(
                [build_denoiser(denoiser_name, **dk) for _ in range(T)]
            )

        # self.log_betas = nn.Parameter(torch.zeros(T))
        self.beta_schedule = build_beta_schedule(
            name=beta_mode, T=T, **(beta_kwargs or {}),
        )

    @torch.no_grad()
    def _compute_targets_on_gpu(
        self, x_gt: torch.Tensor, deltas: torch.Tensor
    ) -> list[torch.Tensor]:
        """Fallback: compute targets on GPU (for trainable schedule).

        Only called when schedule is trainable and deltas are dynamic.
        """
        p = self.pad_border
        B, C, H, W = x_gt.shape
        x_gt_pad = F.pad(x_gt, (p, p, p, p), mode="reflect")
        Hp, Wp = H + 2 * p, W + 2 * p

        targets = [x_gt]  # targets[0] = clean (already on original grid)
        current = x_gt_pad
        for t in range(self.T):
            otf_t = gaussian_otf(deltas[t], Hp, Wp,
                                 device=x_gt.device, dtype=x_gt.dtype)
            current = fft_conv2d_circular(current, otf_t)
            targets.append(current[:, :, p:p+H, p:p+W])
        del x_gt_pad
        return targets

    def forward(
        self,
        y: torch.Tensor,
        sigma: torch.Tensor | float,
        x_gt: torch.Tensor | None = None,
        precomputed_targets: list[torch.Tensor] | None = None,
    ) -> dict:
        """Run T-stage unrolled deblurring.

        Args:
            y:     (B, C, H, W) blurry input
            sigma: scalar or (B,) tensor
            x_gt:  (B, C, H, W) clean image (only needed if no precomputed_targets
                   and schedule is trainable)
            precomputed_targets: list of T+1 tensors (B, C, H, W) from dataset
                   targets[0]=clean, targets[T]=most blurred
                   If provided, skip _compute_targets entirely.

        Returns:
            dict with pred, stage_outputs, stage_targets, deltas
        """
        B, C, H, W = y.shape
        device = y.device
        p = self.pad_border

        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=device, dtype=y.dtype)
        if sigma.dim() > 0:
            sigma = sigma[0]  # use first value (all same in batch)

        # per-stage deltas
        # deltas = self.delta_schedule()  # (T,)
        sigmas = self.delta_schedule(sigma) # (T,)
        betas = self.beta_schedule(sigma, sigmas) # (T,)
        
        # ── Resolve stage targets ───────────────────────────────
        stage_targets = None
        if precomputed_targets is not None:
            # from dataset: targets[0]=clean, ..., targets[T]=most blurred
            # stage T supervises with targets[T-1], ..., stage 1 with targets[0]
            all_targets = precomputed_targets
            stage_targets = [all_targets[s - 1] for s in range(self.T, 0, -1)]
        elif x_gt is not None and self.schedule_name == "trainable":
            # trainable schedule: must recompute with current deltas
            all_targets = self._compute_targets_on_gpu(x_gt, sigmas)
            stage_targets = [all_targets[s - 1] for s in range(self.T, 0, -1)]
            del all_targets

        # ── Reflect-pad blurry input ────────────────────────────
        y_pad = F.pad(y, (p, p, p, p), mode="reflect")
        Hp, Wp = H + 2 * p, W + 2 * p

        # ── Reverse chain on padded grid ────────────────────────
        stage_outputs = []
        x_t = y_pad
        del y_pad

        for idx, s in enumerate(range(self.T, 0, -1)):
            t = s - 1
            sigma_t = sigmas[t]
            beta_t = betas[t]

            otf = gaussian_otf(sigma_t, Hp, Wp, device=device, dtype=y.dtype)

            x_t = self.solver.step(
                x_t,
                denoiser=self.denoisers[t],
                otf=otf,
                beta=beta_t,
                inner_iters=self.inner_iters,
            )
            stage_outputs.append(x_t[:, :, p:p+H, p:p+W])

        final = stage_outputs[-1].clamp(0, 1)

        return {
            "pred": final,
            "stage_outputs": stage_outputs,
            "stage_targets": stage_targets,
            "sigmas": sigmas,
        }