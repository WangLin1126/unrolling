"""PAP utility functions."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only trainable parameters (for optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]


def summarize_chain(model) -> str:
    """Return a human-readable summary of the denoiser chain."""
    lines = []
    lines.append(f"PAP Denoiser Chain (T={model.T}):")
    lines.append("-" * 60)
    for i, (entry, denoiser) in enumerate(zip(model.denoiser_chain_cfg, model.denoisers)):
        n_params = sum(p.numel() for p in denoiser.parameters())
        n_train = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        frozen_str = "FROZEN" if not entry.get("trainable", True) else "trainable"
        pretrain_str = entry.get("pretrain") or "scratch"
        lines.append(
            f"  Stage {i}: {entry['type']:12s} | pos={entry['position']} | "
            f"params={n_params:>8,} ({frozen_str}, {n_train:,} trainable) | "
            f"pretrain={pretrain_str}"
        )
    lines.append("-" * 60)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"  Total: {total:,} params ({trainable:,} trainable)")
    return "\n".join(lines)
