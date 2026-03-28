"""Helpers for stage-wise training of unrolled networks.

Three training modes:
  - end2end:          standard end-to-end backprop (default, no helpers needed)
  - one_then_another: train one denoiser at a time for a fixed epoch budget
  - gradual_in_epoch: T separate forward-backward passes per batch, one per stage
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Helpers for one_then_another ──────────────────────────────────────

def get_active_stage(epoch: int, total_epochs: int, T: int) -> int:
    """Return which denoiser index (0..T-1) is active at *epoch* (1-based)."""
    epochs_per_stage = max(total_epochs // T, 1)
    return min((epoch - 1) // epochs_per_stage, T - 1)


def freeze_denoisers_except(model: nn.Module, active_stage: int):
    """Freeze all denoisers except ``denoisers[active_stage]``."""
    for i, denoiser in enumerate(model.denoisers):
        requires = (i == active_stage)
        for p in denoiser.parameters():
            p.requires_grad = requires


def unfreeze_all_denoisers(model: nn.Module):
    """Restore ``requires_grad=True`` on every denoiser parameter."""
    for denoiser in model.denoisers:
        for p in denoiser.parameters():
            p.requires_grad = True


# ── Helpers for gradual_in_epoch ──────────────────────────────────────

def build_per_stage_optimizers(
    model: nn.Module,
    criterion: nn.Module,
    lr: float,
    weight_decay: float,
    T: int,
) -> list[torch.optim.Optimizer]:
    """Create T AdamW optimisers, one per denoiser.

    Criterion parameters (e.g. learnable loss weights) and any trainable
    schedule parameters that live on the *model* (but outside the denoisers)
    are added to the **last** optimiser so they still receive updates.
    """
    optimizers: list[torch.optim.Optimizer] = []

    # Collect parameter ids that belong to denoisers
    denoiser_param_ids: set[int] = set()
    for denoiser in model.denoisers:
        for p in denoiser.parameters():
            denoiser_param_ids.add(id(p))

    # Non-denoiser model params (schedules, etc.)
    extra_model_params = [
        p for p in model.parameters()
        if id(p) not in denoiser_param_ids and p.requires_grad
    ]
    criterion_params = [p for p in criterion.parameters() if p.requires_grad]

    for t in range(T):
        params = list(model.denoisers[t].parameters())
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
    """Create one LR scheduler per optimiser (mirrors the single-optimizer logic)."""
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
