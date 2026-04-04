"""Plug-and-Play unrolled deblurring with heterogeneous denoiser chain.

Unlike UnrolledDeblurNet which uses the same denoiser architecture for all stages,
PaPDeblurNet supports a different denoiser type/config at each stage, with optional
pretrained weight loading and per-stage freeze control.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.schedule import (
    build_blur_sigma_schedule,
    build_noise_sigma_schedule,
    build_beta_schedule,
)
from models.denoisers import build_denoiser, apply_denoiser
from models.solvers import build_solver
from models.fft_ops import build_blur_operator, fft_conv2d_circular, precompute_freq_sq

logger = logging.getLogger(__name__)


class PaPDeblurNet(nn.Module):
    """Plug-and-Play unrolled deblurring with heterogeneous denoiser chain.

    Args:
        denoiser_chain: sorted list of dicts from parse_denoiser_chain(), each with:
            type, position, pretrain, trainable, params
        solver_name: solver type
        in_channels: image channels
        pad_border: reflect-pad pixels for FFT
        blur_sigma_schedule / kwargs: blur decomposition schedule
        beta_schedule / beta_kwargs: regularization weight schedule
        noise_sigma_schedule / kwargs: noise level schedule
        inner_iters: solver iterations per stage
    """

    def __init__(
        self,
        denoiser_chain: list[dict],
        solver_name: str = "hqs",
        in_channels: int = 3,
        pad_border: int = 32,
        blur_sigma_schedule: str = "uniform",
        blur_sigma_schedule_kwargs: dict | None = None,
        beta_schedule: str = "geom",
        beta_kwargs: dict | None = None,
        noise_sigma_schedule: str = "loguniform",
        noise_sigma_schedule_kwargs: dict | None = None,
        inner_iters: int = 1,
        kernel_size: int = -1,
    ):
        super().__init__()
        T = len(denoiser_chain)
        self.T = T
        self.inner_iters = inner_iters
        self.pad_border = pad_border
        self.kernel_size = kernel_size
        self.blur_sigma_schedule_name = blur_sigma_schedule
        self.denoiser_chain_cfg = denoiser_chain

        # Schedules
        self.blur_sigma_schedule = build_blur_sigma_schedule(
            blur_sigma_schedule, T=T, **(blur_sigma_schedule_kwargs or {}),
        )
        self.noise_sigma_schedule = build_noise_sigma_schedule(
            noise_sigma_schedule, T=T, **(noise_sigma_schedule_kwargs or {}),
        )
        self.beta_schedule_name = beta_schedule
        self.beta_schedule = build_beta_schedule(
            name=beta_schedule, T=T, **(beta_kwargs or {}),
        )

        # Solver
        self.solver = build_solver(solver_name)

        # Build heterogeneous denoiser chain
        self.denoisers = nn.ModuleList()
        for i, entry in enumerate(denoiser_chain):
            params = dict(entry["params"])
            params["in_channels"] = in_channels
            denoiser = build_denoiser(entry["type"], **params)

            # Load pretrained weights
            if entry.get("pretrain") is not None:
                ckpt_path = Path(entry["pretrain"])
                logger.info(f"Stage {i} ({entry['type']}): loading pretrain from {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model" in ckpt:
                    # Extract stage-specific weights from full model checkpoint
                    state = _extract_denoiser_weights(ckpt["model"], i)
                    if state:
                        denoiser.load_state_dict(state, strict=False)
                    else:
                        # Try loading as standalone denoiser checkpoint
                        denoiser.load_state_dict(ckpt["model"], strict=False)
                elif isinstance(ckpt, dict):
                    denoiser.load_state_dict(ckpt, strict=False)
                else:
                    denoiser.load_state_dict(ckpt, strict=False)

            # Freeze if not trainable
            if not entry.get("trainable", True):
                for param in denoiser.parameters():
                    param.requires_grad = False
                denoiser.eval()
                logger.info(f"Stage {i} ({entry['type']}): frozen (trainable=false)")

            self.denoisers.append(denoiser)

    @torch.no_grad()
    def _compute_targets_on_gpu(
        self, x_gt: torch.Tensor, blur_sigma_deltas: torch.Tensor
    ) -> list[torch.Tensor]:
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
            targets.append(current[:, :, p : p + H, p : p + W])
        del x_gt_pad
        return targets

    def forward(
        self,
        blur: torch.Tensor,
        blur_sigma: torch.Tensor | float,
        noise_sigma: torch.Tensor | float,
        x_gt: torch.Tensor | None = None,
        precomputed_targets: list[torch.Tensor] | None = None,
    ) -> dict:
        B, C, H, W = blur.shape
        device = blur.device
        p = self.pad_border

        if not isinstance(blur_sigma, torch.Tensor):
            blur_sigma = torch.tensor(blur_sigma, device=device, dtype=blur.dtype)
        if not isinstance(noise_sigma, torch.Tensor):
            noise_sigma = torch.tensor(noise_sigma, device=device, dtype=blur.dtype)
        if blur_sigma.dim() == 0:
            blur_sigma = blur_sigma.unsqueeze(0).expand(B)
        if noise_sigma.dim() == 0:
            noise_sigma = noise_sigma.unsqueeze(0).expand(B)

        # Per-stage schedules
        blur_sigma_deltas = self.blur_sigma_schedule(blur_sigma)
        noise_sigma_levels = self.noise_sigma_schedule(noise_sigma)
        if self.beta_schedule_name == "dpir":
            betas = self.beta_schedule(noise_sigma, noise_sigma_levels)
        else:
            betas = self.beta_schedule(blur_sigma, blur_sigma_deltas)

        # Resolve stage targets
        stage_targets = None
        if precomputed_targets is not None:
            all_targets = precomputed_targets
            stage_targets = [all_targets[s - 1] for s in range(self.T, 0, -1)]
        elif x_gt is not None and self.blur_sigma_schedule_name == "trainable":
            all_targets = self._compute_targets_on_gpu(x_gt, blur_sigma_deltas)
            stage_targets = [all_targets[s - 1] for s in range(self.T, 0, -1)]
            del all_targets

        # Reflect-pad blurry input
        blur_pad = F.pad(blur, (p, p, p, p), mode="reflect")
        Hp, Wp = H + 2 * p, W + 2 * p
        freq_sq = precompute_freq_sq(Hp, Wp, device, blur.dtype)

        # Reverse chain
        stage_outputs = []
        x_t = blur_pad
        del blur_pad

        for t in range(self.T):
            blur_sigma_t = blur_sigma_deltas[:, self.T - 1 - t]
            beta_t = betas[:, self.T - 1 - t]
            noise_sigma_t = noise_sigma_levels[:, self.T - 1 - t]
            otf = build_blur_operator(
                blur_sigma_t, Hp, Wp,
                kernel_size=self.kernel_size,
                device=device, dtype=blur.dtype,
                freq_sq=freq_sq,
            ).otf

            x_t = self.solver.step(
                x_t,
                denoiser=self.denoisers[t],
                otf=otf,
                beta=beta_t,
                noise_sigma=noise_sigma_t,
                inner_iters=self.inner_iters,
            )
            stage_outputs.append(x_t[:, :, p : p + H, p : p + W])

        final = stage_outputs[-1].clamp(0, 1)

        return {
            "pred": final,
            "stage_outputs": stage_outputs,
            "stage_targets": stage_targets,
            "blur_sigma_deltas": blur_sigma_deltas,
        }

    def train(self, mode: bool = True):
        """Override to keep frozen denoisers in eval mode."""
        super().train(mode)
        if mode:
            for i, entry in enumerate(self.denoiser_chain_cfg):
                if not entry.get("trainable", True):
                    self.denoisers[i].eval()
        return self


def _extract_denoiser_weights(
    full_state_dict: dict, stage_idx: int
) -> dict | None:
    """Try to extract denoiser weights for a specific stage from a full model checkpoint."""
    prefix = f"denoisers.{stage_idx}."
    extracted = {}
    for k, v in full_state_dict.items():
        if k.startswith(prefix):
            extracted[k[len(prefix):]] = v
    return extracted if extracted else None
