"""Gradually-freeze: start training all denoisers, progressively freeze earlier ones.

Phase layout for T stages (T+1 equal phases within ``total_epochs``):
  phase 0  (epochs 1 … E/(T+1)):       all denoisers trainable
  phase 1  (epochs E/(T+1)+1 … 2E/(T+1)):  freeze denoiser 0
  phase 2  :                              freeze denoiser 0,1
  …
  phase T  :                              freeze denoiser 0..T-2
                                          (only denoiser T-1 still trains)

The last denoiser is *never* frozen so training is always meaningful.
"""

from __future__ import annotations

import torch

from .common import (
    TrainContext,
    forward_model,
    compute_criterion_loss,
    compute_single_stage_loss,
    freeze_denoisers_up_to,
    unfreeze_all_denoisers,
    get_freeze_boundary,
    unwrap_model,
    autocast_context,
    amp_backward,
    amp_optimizer_step,
)


def setup_gradually_freeze(ctx: TrainContext, epoch: int) -> int:
    """Freeze denoisers according to the current phase.

    Returns:
        last_frozen: index of the last frozen denoiser (-1 = all trainable).
    """
    tc = ctx.cfg["train"]
    last_frozen = get_freeze_boundary(epoch, tc["epochs"], ctx.T)
    raw_model = unwrap_model(ctx.model)
    if last_frozen < 0:
        unfreeze_all_denoisers(raw_model)
    else:
        freeze_denoisers_up_to(raw_model, last_frozen)
    return last_frozen


def train_one_epoch_gradually_freeze(
    ctx: TrainContext,
    train_loader,
    epoch: int,
    last_frozen: int,
) -> float:
    """One epoch of gradually-freeze training.

    Uses the full StagewiseLoss criterion (same as end2end) but with earlier
    denoisers progressively frozen so their parameters stop receiving
    gradient updates.
    """
    tc = ctx.cfg["train"]
    train_loss_sum = 0.0
    train_count = 0

    for step, (blur, sharp, blur_sigmas, noise_sigmas, targets) in enumerate(train_loader, 1):
        blur = blur.to(ctx.device, non_blocking=True)
        sharp = sharp.to(ctx.device, non_blocking=True)
        blur_sigmas = blur_sigmas.to(ctx.device, non_blocking=True)
        noise_sigmas = noise_sigmas.to(ctx.device, non_blocking=True)
        targets_gpu = (
            [t.to(ctx.device, non_blocking=True) for t in targets]
            if ctx.use_precomputed else None
        )

        # Full forward pass through all T stages.  Frozen denoisers still
        # execute (needed for correct input to later stages) but their
        # parameters don't accumulate gradients.
        with autocast_context(ctx):
            result = forward_model(ctx, blur, blur_sigmas, noise_sigmas, sharp, targets_gpu)
            loss, info = compute_criterion_loss(ctx, result, sharp, blur, blur_sigmas)

        ctx.optimizer.zero_grad(set_to_none=True)
        amp_backward(ctx, loss)
        amp_optimizer_step(ctx, ctx.optimizer, ctx.all_params, tc["grad_clip"])

        bs = blur.shape[0]
        train_loss_sum += loss.item() * bs
        train_count += bs

        if ctx.logger and step % tc["log_every"] == 0:
            frozen_str = f"frozen≤{last_frozen}" if last_frozen >= 0 else "all_trainable"
            w_str = ", ".join(f"{w:.3f}" for w in info["weights"])
            ctx.logger.info(
                f"[Train] E{epoch} S{step} "
                f"loss={loss.item():.5f} ({frozen_str}) "
                f"stage_losses={[f'{l:.4f}' for l in info['per_stage_loss']]} "
                f"weights=[{w_str}]"
            )

    return train_loss_sum / max(train_count, 1)
