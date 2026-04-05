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

from .schedule import build_blur_sigma_schedule, build_noise_sigma_schedule, build_beta_schedule
from .denoisers import build_denoiser, apply_denoiser
from .solvers import build_solver
from .fft_ops import build_blur_operator, fft_conv2d_circular, precompute_freq_sq


class UnrolledDeblurNet(nn.Module):
    """Unrolled multi-stage Gaussian deblurring with intermediate supervision.

    Args:
        T:              number of unrolling stages
        solver_name:    "hqs" | "admm" | "pg"
        blur_sigma_schedule:  "uniform" | "trainable" | "geom" | "power"
        denoiser_name:  "dncnn" | "unet" | "resblock" | "drunet"
        share_denoisers: share one denoiser across all stages
        inner_iters:    inner solver iterations per stage
        in_channels:    image channels
        pad_border:     reflect-pad pixels for FFT (MUST match dataset)
        denoiser_kwargs: kwargs for denoiser construction
        blur_sigma_schedule_kwargs: kwargs for blur sigma schedule
        beta_schedule:  "constant" | "geom" | "geom_dec" | "dpir"
        beta_kwargs:    kwargs for beta schedule
        noise_sigma_schedule:  "loguniform"
        noise_sigma_schedule_kwargs: kwargs for noise sigma schedule
    """

    def __init__(
        self,
        T: int = 5,
        solver_name: str = "hqs",
        blur_sigma_schedule: str = "uniform",
        denoiser_name: str = "dncnn",
        share_denoisers: bool = False,
        inner_iters: int = 1,
        in_channels: int = 3,
        pad_border: int = 32,
        denoiser_kwargs: dict | None = None,
        blur_sigma_schedule_kwargs: dict | None = None,
        beta_kwargs: dict | None = None,
        beta_schedule: str = "geom",
        noise_sigma_schedule: str = "loguniform",
        noise_sigma_schedule_kwargs: dict | None = None,
        kernel_size: int = -1,
        use_pre_denoiser: bool = False,
    ):
        super().__init__()
        self.T = T
        self.inner_iters = inner_iters
        self.pad_border = pad_border
        self.kernel_size = kernel_size
        self.blur_sigma_schedule_name = blur_sigma_schedule
        self.blur_sigma_schedule = build_blur_sigma_schedule(
            blur_sigma_schedule, T=T, **(blur_sigma_schedule_kwargs or {}),
        )
        self.noise_sigma_schedule = build_noise_sigma_schedule(
            noise_sigma_schedule, T=T, **(noise_sigma_schedule_kwargs or {}),
        )
        self.solver = build_solver(solver_name)

        denoiser_cfg = denoiser_kwargs.get(denoiser_name, {})
        dk = dict(in_channels=in_channels, **denoiser_cfg)
        if share_denoisers:
            single = build_denoiser(denoiser_name, **dk)
            self.denoisers = nn.ModuleList([single] * T)
        else:
            self.denoisers = nn.ModuleList(
                [build_denoiser(denoiser_name, **dk) for _ in range(T)]
            )
        self.beta_schedule_name = beta_schedule
        self.beta_schedule = build_beta_schedule(
            name=beta_schedule, T=T, **(beta_kwargs or {}),
        )

        # Optional pre-denoiser: denoises the noisy blurred input before
        # the unrolling stages.  Its loss is computed against G_sigma * x
        # (noise-free blur).  Shares freeze state with denoisers[0].
        if use_pre_denoiser:
            self.pre_denoiser = build_denoiser(denoiser_name, **dk)
        else:
            self.pre_denoiser = None

    @torch.no_grad()
    def _compute_targets_on_gpu(
        self, x_gt: torch.Tensor, blur_sigma_deltas: torch.Tensor
    ) -> list[torch.Tensor]:
        """Fallback: compute targets on GPU (for trainable schedule).

        Args:
            blur_sigma_deltas: (B, T) per-sample per-stage blur stds
        """
        p = self.pad_border
        B, C, H, W = x_gt.shape
        x_gt_pad = F.pad(x_gt, (p, p, p, p), mode="reflect")
        Hp, Wp = H + 2 * p, W + 2 * p
        freq_sq = precompute_freq_sq(Hp, Wp, x_gt.device, x_gt.dtype)

        targets = [x_gt]
        current = x_gt_pad
        for t in range(self.T):
            otf_t = build_blur_operator(
                blur_sigma_deltas[:, t], Hp, Wp,
                kernel_size=self.kernel_size,
                device=x_gt.device, dtype=x_gt.dtype,
                freq_sq=freq_sq,
            ).otf
            current = fft_conv2d_circular(current, otf_t)
            targets.append(current[:, :, p:p+H, p:p+W])
        del x_gt_pad
        return targets

    def forward(
        self,
        blur: torch.Tensor,
        blur_sigma: torch.Tensor | float,
        noise_sigma: torch.Tensor | float,
        x_gt: torch.Tensor | None = None,
        precomputed_targets: list[torch.Tensor] | None = None,
        max_stage: int | None = None,
        active_stage: int | None = None,
        detach_between_stages: bool = False,
        blur_clean: torch.Tensor | None = None,
    ) -> dict:
        """Run T-stage unrolled deblurring.

        Args:
            blur:       (B, C, H, W) blurry input
            blur_sigma: scalar or (B,) per-sample blur std
            noise_sigma: scalar or (B,) per-sample noise std
            x_gt:       (B, C, H, W) clean image (trainable schedule only)
            precomputed_targets: list of T+1 tensors from dataset
            max_stage:  only run stages 0..max_stage (default: T-1)
            active_stage: only this denoiser receives gradients; earlier
                          stages run under ``torch.no_grad()`` and the
                          input is detached at the boundary (default: None,
                          meaning all stages are active / end-to-end)
            detach_between_stages: if True, detach ``x_t`` before every
                          stage t>0 so that later-stage losses cannot
                          back-propagate into earlier denoisers

        Returns:
            dict with pred, stage_outputs, stage_targets, blur_sigma_deltas
        """
        B, C, H, W = blur.shape
        device = blur.device
        p = self.pad_border

        if not isinstance(blur_sigma, torch.Tensor):
            blur_sigma = torch.tensor(blur_sigma, device=device, dtype=blur.dtype)
        if not isinstance(noise_sigma, torch.Tensor):
            noise_sigma = torch.tensor(noise_sigma, device=device, dtype=blur.dtype)
        # Ensure at least 1-dim for consistent (B,) shape through schedules
        if blur_sigma.dim() == 0:
            blur_sigma = blur_sigma.unsqueeze(0).expand(B)
        if noise_sigma.dim() == 0:
            noise_sigma = noise_sigma.unsqueeze(0).expand(B)

        # Per-stage schedules: all (B, T)
        blur_sigma_deltas = self.blur_sigma_schedule(blur_sigma)
        noise_sigma_levels = self.noise_sigma_schedule(noise_sigma)
        if self.beta_schedule_name == 'dpir':
            betas = self.beta_schedule(noise_sigma, noise_sigma_levels)
        else:
            betas = self.beta_schedule(blur_sigma, blur_sigma_deltas)

        # ── Resolve stage targets ───────────────────────────────
        stage_targets = None
        if precomputed_targets is not None:
            all_targets = precomputed_targets
            stage_targets = [all_targets[s - 1] for s in range(self.T, 0, -1)] # transform to stage_targets[-1] is the clear image
        elif x_gt is not None and self.blur_sigma_schedule_name == "trainable":
            all_targets = self._compute_targets_on_gpu(x_gt, blur_sigma_deltas)
            stage_targets = [all_targets[s - 1] for s in range(self.T, 0, -1)]
            del all_targets

        # ── Reflect-pad blurry input ────────────────────────────
        blur_pad = F.pad(blur, (p, p, p, p), mode="reflect")
        Hp, Wp = H + 2 * p, W + 2 * p
        freq_sq = precompute_freq_sq(Hp, Wp, device, blur.dtype)

        # ── Pre-denoiser: denoise noisy blur before unrolling ───
        pre_denoiser_output = None
        pre_denoiser_target = None
        if self.pre_denoiser is not None:
            blur_pad = apply_denoiser(self.pre_denoiser, blur_pad, noise_sigma)
            pre_denoiser_output = blur_pad[:, :, p:p+H, p:p+W]
            pre_denoiser_target = blur_clean

        # ── Reverse chain on padded grid ────────────────────────
        effective_T = (max_stage + 1) if max_stage is not None else self.T

        stage_outputs = []
        x_t = blur_pad
        del blur_pad

        for t in range(effective_T):
            blur_sigma_t = blur_sigma_deltas[:, self.T-1-t]         # (B,)
            beta_t = betas[:, self.T-1-t]                           # (B,)
            noise_sigma_t = noise_sigma_levels[:, self.T-1-t]       # (B,)
            otf = build_blur_operator(
                blur_sigma_t, Hp, Wp,
                kernel_size=self.kernel_size,
                device=device, dtype=blur.dtype,
                freq_sq=freq_sq,
            ).otf

            if active_stage is not None and t < active_stage:
                # Earlier stages: no gradient computation
                with torch.no_grad():
                    x_t = self.solver.step(
                        x_t,
                        denoiser=self.denoisers[t],
                        otf=otf,
                        beta=beta_t,
                        noise_sigma=noise_sigma_t,
                        inner_iters=self.inner_iters,
                    )
            else:
                if active_stage is not None and t == active_stage:
                    # Detach to cut gradient flow from earlier stages
                    x_t = x_t.detach()
                if detach_between_stages and t > 0:
                    # Stage-wise detached: block gradient from stage t's
                    # loss back into stages 0..t-1
                    x_t = x_t.detach()
                x_t = self.solver.step(
                    x_t,
                    denoiser=self.denoisers[t],
                    otf=otf,
                    beta=beta_t,
                    noise_sigma=noise_sigma_t,
                    inner_iters=self.inner_iters,
                )
            stage_outputs.append(x_t[:, :, p:p+H, p:p+W])

        final = stage_outputs[-1].clamp(0, 1)

        # Filter stage_targets to match effective_T length
        if stage_targets is not None and max_stage is not None:
            stage_targets = stage_targets[:effective_T]

        return {
            "pred": final,
            "stage_outputs": stage_outputs,
            "stage_targets": stage_targets,
            "blur_sigma_deltas": blur_sigma_deltas,
            "pre_denoiser_output": pre_denoiser_output,
            "pre_denoiser_target": pre_denoiser_target,
        }
