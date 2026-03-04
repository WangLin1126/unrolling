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

class GeomBetaSchedule(nn.Module):
    """
    beta_t = beta_min * (beta_max/beta_min)^(t/(T-1))   t=0..T-1
    optionally scaled by noise sigma.
    """
    def __init__(self, T: int, beta_min: float, beta_max: float, scale_by: str = "none"):
        super().__init__()
        self.T = T
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        assert scale_by in ["none", "inv_sigma2", "sigma2"]
        self.scale_by = scale_by

    def forward(self, sigma: torch.Tensor):
        # sigma: scalar tensor
        device = sigma.device
        dtype = sigma.dtype

        if self.T == 1:
            base = torch.tensor([self.beta_max], device=device, dtype=dtype)
        else:
            t = torch.linspace(0, 1, self.T, device=device, dtype=dtype)
            base = self.beta_min * (self.beta_max / self.beta_min) ** t  # (T,)

        if self.scale_by == "none":
            return base
        elif self.scale_by == "inv_sigma2":
            return base / (sigma * sigma + 1e-12)
        else:  # "sigma2"
            return base * (sigma * sigma)

def build_beta_schedule(name: str, T: int, beta_min: float, beta_max: float, scale_by: str = "none"):
    if name == "geom":
        return GeomBetaSchedule(T=T, beta_min=beta_min, beta_max=beta_max, scale_by=scale_by)
    raise ValueError(f"Unknown beta schedule: {name}")