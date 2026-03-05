"""Sigma decomposition schedules.

Given total blur σ and T stages, produce {δ_t} such that Σ δ_t² = σ².
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn

_EPS = 1e-12
class UniformSchedule(nn.Module):
    def __init__(self, T: int, **kwargs):
        super().__init__()
        self.T = T

    def forward(self, sigma: torch.Tensor = 1.0) -> torch.Tensor:
        delta = sigma / math.sqrt(self.T)
        return delta.expand(self.T)


class TrainableSchedule(nn.Module):
    """Learnable decomposition: α_t = softmax(w)_t, δ_t = σ·√α_t."""

    def __init__(self, T: int, init: str = "uniform", **kwargs):
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

    def forward(self, sigma: torch.Tensor = 1.0) -> torch.Tensor:
        alpha = torch.softmax(self.logits, dim=0)
        return sigma * torch.sqrt(alpha)

class GeomSchedule(nn.Module):
    """
    alpha_t ∝ r^(T-1-t)  (dec=True, front-heavy)
    delta_t = sigma * sqrt(alpha_t)
    """
    def __init__(self, T: int, r: float = 0.8, front_heavy: bool = True, **kwargs):
        super().__init__()
        self.T = T
        self.r = float(r)
        self.front_heacy = bool(front_heavy)

    def forward(self, sigma: torch.Tensor = 1.0) -> torch.Tensor:
        device, dtype = sigma.device, sigma.dtype
        t = torch.arange(self.T, device=device, dtype=dtype)
        if self.front_heacy:
            w = (self.r ** t)
        else:
            w = (self.r ** (self.T - 1 - t))
        alpha = w / (w.sum() + _EPS)
        return sigma * torch.sqrt(alpha)


class PowerSchedule(nn.Module):
    """
    alpha_t ∝ (T-t)^p  (front_heavy=True)
    delta_t = sigma * sqrt(alpha_t)
    """
    def __init__(self, T: int, p: float = 2.0, front_heavy: bool = True, **kwargs):
        super().__init__()
        self.T = T
        self.p = float(p)
        self.front_heavy = bool(front_heavy)

    def forward(self, sigma: torch.Tensor = 1.0) -> torch.Tensor:
        device, dtype = sigma.device, sigma.dtype
        t = torch.arange(self.T, device=device, dtype=dtype)
        if self.front_heavy:
            w = (self.T - t) ** self.p
        else:
            w = (t + 1) ** self.p
        alpha = w / (w.sum() + 1e-12)
        return sigma * torch.sqrt(alpha)

SCHEDULE_REGISTRY: dict[str, type] = {
    "uniform": UniformSchedule,
    "trainable": TrainableSchedule,
    "geom": GeomSchedule,
    "power": PowerSchedule,
}


def build_schedule(name: str, T: int, **kwargs) -> nn.Module:
    if name not in SCHEDULE_REGISTRY:
        raise ValueError(f"Unknown schedule '{name}'. Choose from {list(SCHEDULE_REGISTRY)}")
    return SCHEDULE_REGISTRY[name](T=T, **kwargs)

def _maybe_scale_by_noise(base: torch.Tensor, sigma: torch.Tensor, scale_by: str):
    """
    base: (T,)
    sigma: scalar tensor
    """
    if scale_by == "none":
        return base
    if scale_by == "sigma2":
        return base * (sigma * sigma)
    if scale_by == "inv_sigma2":
        return base / (sigma * sigma + _EPS)
    raise ValueError(f"Unknown scale_by: {scale_by}")

class ConstantBetaSchedule(nn.Module):
    def __init__(self, T: int, beta: float, scale_by: str = "none"):
        super().__init__()
        self.T = T
        self.beta = float(beta)
        self.scale_by = scale_by

    def forward(self, sigma: torch.Tensor, deltas: torch.Tensor | None = None):
        base = torch.full((self.T,), self.beta, device=sigma.device, dtype=sigma.dtype)
        return _maybe_scale_by_noise(base, sigma, self.scale_by)

class GeomBetaSchedule(nn.Module):
    """
    beta_t = beta_min * (beta_max/beta_min)^(t/(T-1))   if increasing
    beta_t = beta_max * (beta_min/beta_max)^(t/(T-1))   if decreasing
    """
    def __init__(self, T: int, beta_min: float, beta_max: float, decreasing: bool = False, scale_by: str = "none"):
        super().__init__()
        self.T = T
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.decreasing = decreasing
        self.scale_by = scale_by

    def forward(self, sigma: torch.Tensor, deltas: torch.Tensor | None = None):
        device, dtype = sigma.device, sigma.dtype
        if self.T == 1:
            base = torch.tensor([self.beta_max], device=device, dtype=dtype)
        else:
            t = torch.linspace(0, 1, self.T, device=device, dtype=dtype)
            if not self.decreasing:
                base = self.beta_min * (self.beta_max / self.beta_min) ** t
            else:
                base = self.beta_max * (self.beta_min / self.beta_max) ** t
        return _maybe_scale_by_noise(base, sigma, self.scale_by)

class DeltaPowerBetaSchedule(nn.Module):
    """
    beta_t = beta_min * (delta_t / delta_min)^p, then clipped to [beta_min, beta_max]
    Requires deltas (T,).
    """
    def __init__(self, T: int, beta_min: float, beta_max: float, p: float = 2.0, scale_by: str = "none"):
        super().__init__()
        self.T = T
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.p = float(p)
        self.scale_by = scale_by

    def forward(self, sigma: torch.Tensor, deltas: torch.Tensor | None = None):
        assert deltas is not None, "DeltaPowerBetaSchedule requires deltas (T,)"
        # normalize by min delta
        dmin = torch.min(deltas).clamp_min(_EPS)
        base = self.beta_min * (deltas / dmin).clamp_min(_EPS) ** self.p
        base = torch.clamp(base, min=self.beta_min, max=self.beta_max)
        return _maybe_scale_by_noise(base.to(sigma.dtype), sigma, self.scale_by)

class DeltaInterpBetaSchedule(nn.Module):
    """
    Linear interpolation in delta:
      beta = beta_min + (beta_max-beta_min)*norm_delta
    norm_delta = (delta - dmin)/(dmax-dmin)
    Requires deltas (T,).
    """
    def __init__(self, T: int, beta_min: float, beta_max: float, scale_by: str = "none"):
        super().__init__()
        self.T = T
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.scale_by = scale_by

    def forward(self, sigma: torch.Tensor, deltas: torch.Tensor | None = None):
        assert deltas is not None, "DeltaInterpBetaSchedule requires deltas (T,)"
        dmin = torch.min(deltas)
        dmax = torch.max(deltas)
        norm = (deltas - dmin) / (dmax - dmin + _EPS)
        base = self.beta_min + (self.beta_max - self.beta_min) * norm
        return _maybe_scale_by_noise(base.to(sigma.dtype), sigma, self.scale_by)

def build_beta_schedule(
    name: str,
    T: int,
    beta: float = 8.0,
    beta_min: float = 0.5,
    beta_max: float = 64.0,
    p: float = 2.0,
    decreasing: bool = False,
    scale_by: str = "none",
):
    """
    name:
      - "constant"
      - "geom"
      - "geom_inc"
      - "geom_dec"
      - "delta_power"   (needs deltas)
      - "delta_interp"  (needs deltas)
    scale_by:
      - "none" | "sigma2" | "inv_sigma2"
    """
    name = name.lower()
    if name == "constant":
        return ConstantBetaSchedule(T=T, beta=beta, scale_by=scale_by)

    if name in ["geom", "geom_inc"]:
        return GeomBetaSchedule(T=T, beta_min=beta_min, beta_max=beta_max, decreasing=False, scale_by=scale_by)

    if name in ["geom_dec"]:
        return GeomBetaSchedule(T=T, beta_min=beta_min, beta_max=beta_max, decreasing=True, scale_by=scale_by)

    if name == "delta_power":
        return DeltaPowerBetaSchedule(T=T, beta_min=beta_min, beta_max=beta_max, p=p, scale_by=scale_by)

    if name == "delta_interp":
        return DeltaInterpBetaSchedule(T=T, beta_min=beta_min, beta_max=beta_max, scale_by=scale_by)

    raise ValueError(f"Unknown beta schedule: {name}")