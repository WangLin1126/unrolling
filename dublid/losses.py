"""Loss functions for DUBLID blind deblurring.

Implements `compute_cost` (missing from reference code) and `BlindDeblurLoss`
which wraps image + kernel losses following the unrolling repo's loss patterns.

TODO (v2): Add shift-alignment for kernel loss to handle the shift ambiguity
  inherent in blind deconvolution. Currently, kernel loss uses direct MSE
  without alignment, which may penalize correct-but-shifted kernel estimates.
  This simplification is noted in the paper as a training detail that can
  affect convergence but is not essential for the core algorithm.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.losses import build_combined_loss, CharbonnierLoss


class BlindDeblurLoss(nn.Module):
    """Combined image reconstruction + kernel estimation loss.

    total = image_loss(pred, gt) + kappa * kernel_loss(k_pred, k_gt) + alpha * weight_decay

    Args:
        image_loss: loss module for image reconstruction
        kappa:      weight for kernel estimation loss
        alpha:      weight decay coefficient for conv filter weights
        kernel_loss_type: "l2" or "l1" for kernel comparison

    TODO: Implement shift-alignment for kernel loss (see module docstring).
    """

    def __init__(
        self,
        image_loss: nn.Module | None = None,
        kappa: float = 100.0,
        alpha: float = 0.01,
        kernel_loss_type: str = "l2",
    ):
        super().__init__()
        self.image_loss = image_loss or nn.MSELoss()
        self.kappa = kappa
        self.alpha = alpha
        if kernel_loss_type == "l2":
            self.kernel_loss_fn = nn.MSELoss()
        elif kernel_loss_type == "l1":
            self.kernel_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown kernel_loss_type: {kernel_loss_type}")

    def forward(
        self,
        image_pred: torch.Tensor,
        image_gt: torch.Tensor,
        kernel_pred: torch.Tensor,
        kernel_gt: torch.Tensor,
        weight_list: nn.ParameterList | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total blind deblurring loss.

        Args:
            image_pred:  (N, C, H, W) predicted sharp image
            image_gt:    (N, C, H, W) ground truth sharp image
            kernel_pred: (N, Hk, Wk) predicted blur kernel
            kernel_gt:   (N, Hk, Wk) ground truth blur kernel
            weight_list: optional conv filter weights for weight decay

        Returns:
            total_loss, info dict with component losses
        """
        l_image = self.image_loss(image_pred, image_gt)
        l_kernel = self.kernel_loss_fn(kernel_pred, kernel_gt)

        l_wd = torch.tensor(0.0, device=image_pred.device)
        if weight_list is not None and self.alpha > 0:
            for w in weight_list:
                l_wd = l_wd + torch.sum(w ** 2)

        total = l_image + self.kappa * l_kernel + self.alpha * l_wd

        info = {
            "image_loss": l_image.item(),
            "kernel_loss": l_kernel.item(),
            "weight_decay": l_wd.item(),
            "total_loss": total.item(),
        }
        return total, info


def compute_cost(
    image_pred: torch.Tensor,
    image_gt: torch.Tensor,
    kernel_pred: torch.Tensor,
    kernel_gt: torch.Tensor,
    weight_list: nn.ParameterList | None = None,
    kappa: float = 100.0,
    alpha: float = 0.01,
) -> torch.Tensor:
    """Standalone loss function matching the reference code's missing compute_cost.

    total = MSE(image_pred, image_gt) + kappa * MSE(kernel_pred, kernel_gt)
            + alpha * sum(||w||²)

    TODO: Implement shift-alignment for kernel loss (see module docstring).
    """
    l_image = torch.mean((image_pred - image_gt) ** 2)
    l_kernel = torch.mean((kernel_pred - kernel_gt) ** 2)

    l_wd = torch.tensor(0.0, device=image_pred.device)
    if weight_list is not None and alpha > 0:
        for w in weight_list:
            l_wd = l_wd + torch.sum(w ** 2)

    return l_image + kappa * l_kernel + alpha * l_wd


def build_blind_deblur_loss(cfg: dict) -> BlindDeblurLoss:
    """Build BlindDeblurLoss from config dict.

    Expected config structure:
        loss:
          image_loss: "l2"           # or "charbonnier", "l1", etc.
          kernel_weight: 100.0       # kappa
          weight_decay: 0.01         # alpha
          kernel_loss_type: "l2"     # or "l1"
    """
    loss_cfg = cfg.get("loss", {})

    # Build image loss
    image_loss_name = loss_cfg.get("image_loss", "l2")
    if image_loss_name == "l2":
        image_loss = nn.MSELoss()
    elif image_loss_name == "l1":
        image_loss = nn.L1Loss()
    elif image_loss_name == "charbonnier":
        image_loss = CharbonnierLoss()
    else:
        # Try the unrolling repo's combined loss builder
        image_loss = build_combined_loss([{"name": image_loss_name}])

    return BlindDeblurLoss(
        image_loss=image_loss,
        kappa=loss_cfg.get("kernel_weight", 100.0),
        alpha=loss_cfg.get("weight_decay", 0.01),
        kernel_loss_type=loss_cfg.get("kernel_loss_type", "l2"),
    )
