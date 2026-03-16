"""
Schedule modules for unrolled deblurring.

Three schedule families:
  1. BlurSigmaSchedule  — decompose total blur σ into per-stage deltas {δ_t}
  2. NoiseSigmaSchedule — produce per-stage noise levels for denoisers
  3. BetaSchedule       — produce per-stage regularization weights β_t

All schedules accept batched (B,) input and produce (B, T) output.
Scalar (0-dim) input produces (T,) output.
"""

from __future__ import annotations
import torch
import torch.nn as nn

_EPS = 1e-12

# ═══════════════════════════════════════════════════════════════════════
# Blur Sigma Schedules: blur_sigma (B,) → blur_sigma_deltas (B, T)
# Constraint: Σ δ_t² = σ²  (each δ_t = σ·√α_t where Σα_t = 1)
# ═══════════════════════════════════════════════════════════════════════

class BlurSigmaSchedule(nn.Module):
    """Base class for blur sigma decomposition schedules."""

    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def _compute_alpha(self, device, dtype) -> torch.Tensor:
        """Return (T,) weights that sum to 1."""
        raise NotImplementedError

    def forward(self, blur_sigma: torch.Tensor) -> torch.Tensor:
        """blur_sigma: scalar or (B,) → (T,) or (B, T)"""
        alpha = self._compute_alpha(blur_sigma.device, blur_sigma.dtype)
        return blur_sigma.unsqueeze(-1) * torch.sqrt(alpha)


class UniformBlurSchedule(BlurSigmaSchedule):
    """Equal energy per stage: α_t = 1/T for all t."""

    def __init__(self, T: int):
        super().__init__(T)
        self.register_buffer("_alpha", torch.full((T,), 1.0 / T))

    def _compute_alpha(self, device, dtype):
        return self._alpha.to(dtype=dtype)


class TrainableBlurSchedule(BlurSigmaSchedule):
    """Learnable decomposition: α_t = softmax(logits)_t."""

    def __init__(self, T: int, init: str = "uniform"):
        super().__init__(T)
        if init == "uniform":
            self.logits = nn.Parameter(torch.zeros(T))
        elif init == "linear_inc":
            self.logits = nn.Parameter(torch.linspace(-1, 1, T))
        elif init == "linear_dec":
            self.logits = nn.Parameter(torch.linspace(1, -1, T))
        else:
            raise ValueError(f"Unknown init: {init}")

    def _compute_alpha(self, device, dtype):
        return torch.softmax(self.logits, dim=0)


class GeomBlurSchedule(BlurSigmaSchedule):
    """Geometric weighting: α_t ∝ r^t (front_heavy) or r^(T-1-t)."""

    def __init__(self, T: int, r: float = 0.8, front_heavy: bool = True):
        super().__init__(T)
        self.r = float(r)
        self.front_heavy = bool(front_heavy)
        self.register_buffer("_t_idx", torch.arange(T, dtype=torch.float32))

    def _compute_alpha(self, device, dtype):
        t = self._t_idx.to(dtype=dtype)
        if self.front_heavy:
            # front represent initial resotore precedure of the image, not the blur forward process. 
            # The first sigma use for first blur kernel.
            w = self.r ** (self.T - 1 - t)
        else:
            w = self.r ** t
        return w / (w.sum() + _EPS)


class PowerBlurSchedule(BlurSigmaSchedule):
    """Power-law weighting: α_t ∝ (T-t)^p (front_heavy) or (t+1)^p."""

    def __init__(self, T: int, p: float = 2.0, front_heavy: bool = True):
        super().__init__(T)
        self.p = float(p)
        self.front_heavy = bool(front_heavy)
        self.register_buffer("_t_idx", torch.arange(T, dtype=torch.float32))

    def _compute_alpha(self, device, dtype):
        t = self._t_idx.to(dtype=dtype)
        if self.front_heavy:
            w = (self.T - t) ** self.p
        else:
            w = (t + 1) ** self.p
        return w / (w.sum() + _EPS)


def build_blur_sigma_schedule(
    name: str,
    T: int,
    init: str = "uniform",
    r: float = 0.8,
    p: float = 2.0,
    front_heavy: bool = True,
    **_extra,
) -> BlurSigmaSchedule:
    name = name.lower()
    if name == "uniform":
        return UniformBlurSchedule(T=T)
    if name == "trainable":
        return TrainableBlurSchedule(T=T, init=init)
    if name == "geom":
        return GeomBlurSchedule(T=T, r=r, front_heavy=front_heavy)
    if name == "power":
        return PowerBlurSchedule(T=T, p=p, front_heavy=front_heavy)
    raise ValueError(f"Unknown blur sigma schedule '{name}'")


# ═══════════════════════════════════════════════════════════════════════
# Noise Sigma Schedules: noise_sigma (B,) → noise_sigma_levels (B, T)
# ═══════════════════════════════════════════════════════════════════════

class NoiseSigmaSchedule(nn.Module):
    """Base class for per-stage noise sigma schedules."""

    def __init__(self, T: int):
        super().__init__()
        self.T = T

class LogUniformNoiseSigmaSchedule(NoiseSigmaSchedule):
    """Log-uniform interpolation between max_ratio and min_ratio of noise_sigma."""

    def __init__(self, T: int, max_ratio: float = 1.5, min_ratio: float = 0.1):
        super().__init__(T)
        self.max_ratio = max_ratio
        self.min_ratio = min_ratio
        if T > 1:
            self.register_buffer("_t_idx", torch.arange(T, dtype=torch.float32))

    def forward(self, noise_sigma: torch.Tensor) -> torch.Tensor:
        sigma_max = noise_sigma * self.max_ratio
        sigma_min = noise_sigma * self.min_ratio

        if self.T == 1:
            return sigma_max.unsqueeze(-1)

        t = self._t_idx.to(dtype=noise_sigma.dtype)
        ratio = sigma_max / sigma_min.clamp(min=_EPS)
        return sigma_min.unsqueeze(-1) * ratio.unsqueeze(-1) ** (t / (self.T - 1))


def build_noise_sigma_schedule(
    name: str,
    T: int,
    max_ratio: float = 1.5,
    min_ratio: float = 0.1,
    **_extra,
) -> NoiseSigmaSchedule:
    name = name.lower()
    if name == "loguniform":
        return LogUniformNoiseSigmaSchedule(T=T, max_ratio=max_ratio, min_ratio=min_ratio)
    raise ValueError(f"Unknown noise sigma schedule '{name}'")


# ═══════════════════════════════════════════════════════════════════════
# Beta Schedules: blur_sigma (B,), blur_sigma_deltas (B, T) → betas (B, T)
# ═══════════════════════════════════════════════════════════════════════

class BetaSchedule(nn.Module):
    """Base class for per-stage regularization weight schedules."""

    def __init__(self, T: int):
        super().__init__()
        self.T = T

    def _ensure_batch(self, base: torch.Tensor, blur_sigma: torch.Tensor) -> torch.Tensor:
        """Expand (T,) base to (B, T) when blur_sigma is batched."""
        if blur_sigma.dim() >= 1 and base.dim() == 1:
            return base.unsqueeze(0).expand(blur_sigma.shape[0], -1)
        return base


class ConstantBetaSchedule(BetaSchedule):
    """Constant β across all stages."""

    def __init__(self, T: int, beta: float = 8.0):
        super().__init__(T)
        self.register_buffer("_base", torch.full((T,), beta))

    def forward(self, blur_sigma: torch.Tensor, blur_sigma_deltas: torch.Tensor | None = None):
        base = self._base.to(dtype=blur_sigma.dtype)
        return self._ensure_batch(base, blur_sigma)


class GeomBetaSchedule(BetaSchedule):
    """Geometric interpolation between beta_min and beta_max."""

    def __init__(self, T: int, beta_min: float = 0.5, beta_max: float = 64.0,
                 decreasing: bool = False):
        super().__init__(T)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.decreasing = decreasing
        if T > 1:
            self.register_buffer("_t_frac", torch.linspace(0, 1, T))

    def forward(self, blur_sigma: torch.Tensor, blur_sigma_deltas: torch.Tensor | None = None):
        if self.T == 1:
            base = torch.tensor([self.beta_max], device=blur_sigma.device, dtype=blur_sigma.dtype)
        else:
            t = self._t_frac.to(dtype=blur_sigma.dtype)
            if not self.decreasing:
                base = self.beta_min * (self.beta_max / self.beta_min) ** t
            else:
                base = self.beta_max * (self.beta_min / self.beta_max) ** t
        return self._ensure_batch(base, blur_sigma)


class DpirBetaSchedule(BetaSchedule):
    """DPIR regularization: β_t = λ · noise_σ² / noise_σ_t²."""

    def __init__(self, T: int, lam: float = 0.23):
        super().__init__(T)
        self.lam = float(lam)

    def forward(self, noise_sigma: torch.Tensor, noise_sigma_levels: torch.Tensor | None = None):
        ns = noise_sigma.unsqueeze(-1) if noise_sigma.dim() >= 1 else noise_sigma
        return self.lam * (ns * ns) / (noise_sigma_levels * noise_sigma_levels + _EPS)


def build_beta_schedule(
    name: str,
    T: int,
    beta: float = 8.0,
    beta_min: float = 0.5,
    beta_max: float = 64.0,
    lam: float = 0.23,
    **_extra,
) -> BetaSchedule:
    name = name.lower()
    if name == "constant":
        return ConstantBetaSchedule(T=T, beta=beta)
    if name in ("geom", "geom_inc"):
        return GeomBetaSchedule(T=T, beta_min=beta_min, beta_max=beta_max, decreasing=False)
    if name == "geom_dec":
        return GeomBetaSchedule(T=T, beta_min=beta_min, beta_max=beta_max, decreasing=True)
    if name == "dpir":
        return DpirBetaSchedule(T=T, lam=lam)
    raise ValueError(f"Unknown beta schedule '{name}'")
