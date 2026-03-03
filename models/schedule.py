"""Sigma decomposition schedules.

Given total blur σ and T stages, produce {δ_t} such that Σ δ_t² = σ².
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class UniformSchedule(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        delta = sigma / math.sqrt(self.T)
        return delta.expand(self.T)


class TrainableSchedule(nn.Module):
    """Learnable decomposition: α_t = softmax(w)_t, δ_t = σ·√α_t."""

    def __init__(self, T: int, init: str = "uniform"):
        super().__init__()
        self.T = T
        if init == "uniform":
            self.logits = nn.Parameter(torch.zeros(T))
        elif init == "linear_inc":
            self.logits = nn.Parameter(torch.linspace(-1, 1, T))
        elif init == "linear_dec":
            self.logits = nn.Parameter(torch.linspace(1, -1, T))
        else:
            raise ValueError(f"Unknown init: {init}")

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        alpha = torch.softmax(self.logits, dim=0)
        return sigma * torch.sqrt(alpha)


SCHEDULE_REGISTRY: dict[str, type] = {
    "uniform": UniformSchedule,
    "trainable": TrainableSchedule,
}


def build_schedule(name: str, T: int, **kwargs) -> nn.Module:
    if name not in SCHEDULE_REGISTRY:
        raise ValueError(f"Unknown schedule '{name}'. Choose from {list(SCHEDULE_REGISTRY)}")
    return SCHEDULE_REGISTRY[name](T=T, **kwargs)