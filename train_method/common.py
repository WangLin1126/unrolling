"""Shared utilities for all training strategies.

Contains the ``TrainContext`` dataclass that bundles every object a training
step function needs, plus reusable helpers for validation, checkpointing,
and denoiser freezing / unfreezing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def unwrap_model(m: nn.Module) -> nn.Module:
    if isinstance(m, DDP):
        m = m.module
    # torch.compile wraps in OptimizedModule; unwrap to get original
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m

# ── Freeze / unfreeze helpers ─────────────────────────────────────────

def _set_pre_denoiser_grad(model: nn.Module, requires: bool):
    """Set requires_grad on pre_denoiser if it exists."""
    pre = getattr(model, "pre_denoiser", None)
    if pre is not None:
        for p in pre.parameters():
            p.requires_grad = requires


def freeze_denoisers_except(model: nn.Module, active_stage: int):
    """Freeze all denoisers except ``denoisers[active_stage]``.

    Pre-denoiser shares freeze state with denoisers[0].
    """
    for i, denoiser in enumerate(model.denoisers):
        requires = (i == active_stage)
        for p in denoiser.parameters():
            p.requires_grad = requires
    _set_pre_denoiser_grad(model, active_stage == 0)


def freeze_denoisers_up_to(model: nn.Module, last_frozen: int):
    """Freeze denoisers[0..last_frozen], unfreeze the rest.

    Pre-denoiser shares freeze state with denoisers[0]: frozen when
    last_frozen >= 0.
    """
    for i, denoiser in enumerate(model.denoisers):
        requires = (i > last_frozen)
        for p in denoiser.parameters():
            p.requires_grad = requires
    _set_pre_denoiser_grad(model, last_frozen < 0)


def unfreeze_all_denoisers(model: nn.Module):
    """Restore ``requires_grad=True`` on every denoiser parameter."""
    for denoiser in model.denoisers:
        for p in denoiser.parameters():
            p.requires_grad = True
    _set_pre_denoiser_grad(model, True)


def get_active_stage(epoch: int, total_epochs: int, T: int) -> int:
    """Return which denoiser index (0..T-1) is active at *epoch* (1-based)."""
    epochs_per_stage = max(total_epochs // T, 1)
    return min((epoch - 1) // epochs_per_stage, T - 1)


def get_freeze_boundary(epoch: int, total_epochs: int, T: int) -> int:
    """Return the index of the last frozen denoiser for ``gradually_freeze``.

    Phase layout (T phases, each ``epochs_per_phase`` epochs long):
      phase 0: all denoisers trainable              → returns -1
      phase 1: denoisers[0] frozen                  → returns 0
      phase 2: denoisers[0..1] frozen               → returns 1
      …
      phase T-1: denoisers[0..T-2] frozen           → returns T-2

    The last denoiser is never frozen so training never becomes vacuous.
    """
    epochs_per_phase = max(total_epochs // T, 1)
    phase = min((epoch - 1) // epochs_per_phase, T - 1)
    return phase - 1            # -1 means "nothing frozen yet"


# ── Per-stage optimizer / scheduler builders ──────────────────────────

def build_per_stage_optimizers(
    model: nn.Module,
    criterion: nn.Module,
    lr: float,
    weight_decay: float,
    T: int,
) -> list[torch.optim.Optimizer]:
    """Create T AdamW optimisers, one per denoiser.

    Criterion parameters and non-denoiser model parameters are added to the
    **last** optimiser.
    """
    optimizers: list[torch.optim.Optimizer] = []

    denoiser_param_ids: set[int] = set()
    for denoiser in model.denoisers:
        for p in denoiser.parameters():
            denoiser_param_ids.add(id(p))
    # Pre-denoiser params share optimizer with denoisers[0]
    pre_denoiser_params: list[torch.nn.Parameter] = []
    pre = getattr(model, "pre_denoiser", None)
    if pre is not None:
        for p in pre.parameters():
            denoiser_param_ids.add(id(p))
            pre_denoiser_params.append(p)

    extra_model_params = [
        p for p in model.parameters()
        if id(p) not in denoiser_param_ids and p.requires_grad
    ]
    criterion_params = [p for p in criterion.parameters() if p.requires_grad]

    for t in range(T):
        params = list(model.denoisers[t].parameters())
        if t == 0:
            params = params + pre_denoiser_params
        if t == T - 1:
            params = params + extra_model_params + criterion_params
        optimizers.append(
            torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        )
    return optimizers


def build_per_stage_schedulers(
    optimizers: list[torch.optim.Optimizer],
    scheduler_name: str,
    total_epochs: int,
    step_size: int = 50,
    gamma: float = 0.5,
) -> list[torch.optim.lr_scheduler.LRScheduler] | None:
    if scheduler_name == "cosine":
        return [
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)
            for opt in optimizers
        ]
    elif scheduler_name == "step":
        return [
            torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
            for opt in optimizers
        ]
    return None


# ── TrainContext ──────────────────────────────────────────────────────

@dataclass
class TrainContext:
    """Bundle of every object a training step function needs."""

    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any                       # LRScheduler | None
    all_params: list[torch.nn.Parameter]
    device: torch.device
    cfg: dict
    T: int
    use_precomputed: bool
    use_cats: bool
    use_ddp: bool
    train_dir: Path
    logger: logging.Logger
    channels_last: bool = False

    # Optional: per-stage optimisers for gradual_in_epoch
    optimizers: list[torch.optim.Optimizer] | None = None
    schedulers: list | None = None

    # Tracking state
    best_psnr: float = 0.0
    best_val_loss: float = field(default_factory=lambda: float("inf"))
    no_improve_count: int = 0


# ── Reusable building blocks ─────────────────────────────────────────

def forward_model(ctx: TrainContext, blur, blur_sigmas, noise_sigmas, sharp,
                  targets_gpu, *, max_stage=None, active_stage=None,
                  detach_between_stages=False, blur_clean=None):
    """Run the model forward, handling precomputed vs on-the-fly targets."""
    if ctx.channels_last:
        blur = blur.to(memory_format=torch.channels_last)
        if sharp is not None:
            sharp = sharp.to(memory_format=torch.channels_last)
        if targets_gpu is not None:
            targets_gpu = [t.to(memory_format=torch.channels_last) for t in targets_gpu]
        if blur_clean is not None:
            blur_clean = blur_clean.to(memory_format=torch.channels_last)
    if ctx.use_precomputed:
        return ctx.model(
            blur=blur, blur_sigma=blur_sigmas, noise_sigma=noise_sigmas,
            x_gt=None, precomputed_targets=targets_gpu,
            max_stage=max_stage, active_stage=active_stage,
            detach_between_stages=detach_between_stages,
            blur_clean=blur_clean,
        )
    else:
        return ctx.model(
            blur=blur, blur_sigma=blur_sigmas, noise_sigma=noise_sigmas,
            x_gt=sharp, precomputed_targets=None,
            max_stage=max_stage, active_stage=active_stage,
            detach_between_stages=detach_between_stages,
            blur_clean=blur_clean,
        )


def compute_criterion_loss(ctx: TrainContext, result, sharp, blur, blur_sigmas):
    """Compute loss through the StagewiseLoss criterion.

    If the model returned a pre_denoiser_output, its loss against the
    noise-free blurred target is added with the configured weight.
    """
    if ctx.use_cats:
        loss, info = ctx.criterion(
            result["stage_outputs"], result["stage_targets"],
            x_gt=sharp, blur=blur, blur_sigma=blur_sigmas,
        )
    else:
        loss, info = ctx.criterion(result["stage_outputs"], result["stage_targets"])

    # Pre-denoiser auxiliary loss
    pre_out = result.get("pre_denoiser_output")
    pre_tgt = result.get("pre_denoiser_target")
    if pre_out is not None and pre_tgt is not None:
        raw_crit = unwrap_model(ctx.criterion)
        pre_loss = raw_crit.base_loss(pre_out, pre_tgt)
        w = ctx.cfg["model"].get("pre_denoiser_loss_weight", 1.0)
        loss = loss + w * pre_loss
        info["pre_denoiser_loss"] = pre_loss.item()

    return loss, info


def compute_single_stage_loss(ctx: TrainContext, result):
    """Compute base_loss on the last stage output vs its target."""
    raw_crit = unwrap_model(ctx.criterion)
    return raw_crit.base_loss(
        result["stage_outputs"][-1],
        result["stage_targets"][-1],
    )
