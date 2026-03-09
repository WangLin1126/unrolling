"""Loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Sequence
from models.fft_ops import gaussian_otf, fft_conv2d_circular
# ── Base losses ─────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, C1: float = 0.01**2, C2: float = 0.03**2):
        super().__init__()
        self.C1, self.C2, self.window_size = C1, C2, window_size
        self.register_buffer(
            "_window", self._gaussian_window(window_size, 1.5), persistent=False
        )

    @staticmethod
    def _gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target):
        C = pred.shape[1]
        w = self._window.to(pred.device, pred.dtype).expand(C, -1, -1, -1)
        pad = self.window_size // 2
        mu_x = F.conv2d(pred, w, padding=pad, groups=C)
        mu_y = F.conv2d(target, w, padding=pad, groups=C)
        s_x = F.conv2d(pred ** 2, w, padding=pad, groups=C) - mu_x ** 2
        s_y = F.conv2d(target ** 2, w, padding=pad, groups=C) - mu_y ** 2
        s_xy = F.conv2d(pred * target, w, padding=pad, groups=C) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + self.C1) * (2 * s_xy + self.C2)) / (
            (mu_x ** 2 + mu_y ** 2 + self.C1) * (s_x + s_y + self.C2)
        )
        return 1.0 - ssim.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of multiple losses."""

    def __init__(self, losses: dict[str, tuple[nn.Module, float]]):
        super().__init__()
        self.loss_modules = nn.ModuleDict()
        self.weights = {}
        for name, (mod, w) in losses.items():
            self.loss_modules[name] = mod
            self.weights[name] = w

    def forward(self, pred, target):
        total = 0.0
        for name, mod in self.loss_modules.items():
            total = total + self.weights[name] * mod(pred, target)
        return total


# ── Registry ────────────────────────────────────────────────────────

LOSS_REGISTRY: dict[str, type] = {
    "l1": nn.L1Loss,
    "l2": nn.MSELoss,
    "charbonnier": CharbonnierLoss,
    "ssim": SSIMLoss,
}


def build_loss(name: str, **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)


def build_combined_loss(specs: list[dict]) -> nn.Module:
    if len(specs) == 1:
        return build_loss(specs[0]["name"], **specs[0].get("kwargs", {}))
    losses = {}
    for s in specs:
        losses[s["name"]] = (build_loss(s["name"], **s.get("kwargs", {})), s.get("weight", 1.0))
    return CombinedLoss(losses)


# ── Stagewise loss ──────────────────────────────────────────────────

class StagewiseLoss(nn.Module):
    """Compute weighted sum of per-stage losses.

    loss = sum_t  w_t * base_loss(stage_outputs[t], stage_targets[t])

    Weights are uniform or learnable (softmax-parameterised to stay positive and sum to 1).
    """

    def __init__(self, T: int, 
                 base_loss: nn.Module, 
                 learnable: bool = False,
                 mode: str = "all",
                blur_total_sigma: float = 4.0,
                blur_sigma_list=None,
                 ):
        super().__init__()
        self.T = T
        self.base_loss = base_loss
        self.learnable = learnable
        self.mode = mode 
        if learnable:
            self.logits = nn.Parameter(torch.zeros(T))
        else:
            self.register_buffer("logits", torch.zeros(T))
        if blur_sigma_list is None or blur_sigma_list == "":
            if blur_total_sigma > 0:
                self.blur_sigmas = [
                    blur_total_sigma * math.sqrt(k / T)
                    for k in range(T - 1, -1, -1)
                ]
            else:
                self.blur_sigmas = [0.0] * T
        else:
            if isinstance(blur_sigma_list, str):
                self.blur_sigmas = [float(x) for x in blur_sigma_list.split(",") if x.strip()]
            else:
                self.blur_sigmas = [float(x) for x in blur_sigma_list]

            if len(self.blur_sigmas) != T:
                raise ValueError(
                    f"blur_sigma_list must have length T={T}, but got {len(self.blur_sigmas)}"
                )
    @property
    def weights(self) -> torch.Tensor:
        """(T,) weights that sum to 1."""
        return torch.softmax(self.logits, dim=0)


    def forward(
        self,
        stage_outputs: list[torch.Tensor],
        stage_targets: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            stage_outputs: [est_of_x_{T-1}, est_of_x_{T-2}, ..., est_of_x_0]  len=T
            stage_targets: [x_{T-1},        x_{T-2},        ..., x_0]          len=T

        Returns:
            total_loss, info_dict with per-stage losses and weights
        """
        assert len(stage_outputs) == len(stage_targets) == self.T
        w = self.weights
        total = torch.tensor(0.0, device=stage_outputs[0].device)
        per_stage = []
        for t in range(self.T):
            if self.mode == "last":
                l_t = self.base_loss(stage_outputs[t], stage_targets[-1])
            elif self.mode == "all":
                l_t = self.base_loss(stage_outputs[t], stage_targets[t])
            elif self.mode == "one_stage":
                l_t = self.T * self.base_loss(stage_outputs[t], stage_targets[t]) if t == self.T-1 else torch.zeros((), device=stage_outputs[t].device)
            elif self.mode == "blur_last":
                # pointwise L1 loss map: (B,C,H,W)
                loss_map = (stage_outputs[t] - stage_targets[-1]).abs()
                sigma = self.blur_sigmas[t]
                if sigma > 1e-12:
                    B, C, H, W = loss_map.shape
                    p = max(1, int(math.ceil(3.0 * sigma)))
                    # reflect-pad -> FFT blur -> crop back
                    loss_pad = F.pad(loss_map, (p, p, p, p), mode="reflect")
                    Hp, Wp = H + 2 * p, W + 2 * p
                    otf = gaussian_otf(
                        sigma,
                        Hp,
                        Wp,
                        device=loss_map.device,
                        dtype=loss_map.dtype,
                    )
                    loss_pad = fft_conv2d_circular(loss_pad, otf)
                    loss_map = loss_pad[:, :, p:p+H, p:p+W]

                # l_t = self.base_loss(loss_map, torch.zeros_like(loss_map))
                l_t = loss_map.mean()

            else:
                raise ValueError(f"Unknown StagewiseLoss mode: {self.mode}")
            total = total + w[t] * l_t
            per_stage.append(l_t.item())

        info = {
            "per_stage_loss": per_stage,
            "weights": w.detach().cpu().tolist(),
        }
        return total, info