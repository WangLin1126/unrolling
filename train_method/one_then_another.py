"""One-then-another: train a single denoiser for epochs/T epochs, then advance."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import (
    TrainContext,
    forward_model,
    compute_single_stage_loss,
    freeze_denoisers_except,
    get_active_stage,
    unwrap_model,
)


def setup_one_then_another(ctx: TrainContext, epoch: int) -> int:
    """Freeze/unfreeze denoisers and return the active stage index."""
    tc = ctx.cfg["train"]
    cur_active = get_active_stage(epoch, tc["epochs"], ctx.T)
    raw_model = unwrap_model(ctx.model)
    freeze_denoisers_except(raw_model, cur_active)
    return cur_active


def train_one_epoch_one_then_another(
    ctx: TrainContext,
    train_loader,
    epoch: int,
    active_stage: int,
) -> float:
    tc = ctx.cfg["train"]
    train_loss_sum = 0.0
    train_count = 0

    for step, (blur, sharp, blur_sigmas, noise_sigmas, targets, blur_clean) in enumerate(train_loader, 1):
        blur = blur.to(ctx.device, non_blocking=True)
        sharp = sharp.to(ctx.device, non_blocking=True)
        blur_sigmas = blur_sigmas.to(ctx.device, non_blocking=True)
        noise_sigmas = noise_sigmas.to(ctx.device, non_blocking=True)
        blur_clean = blur_clean.to(ctx.device, non_blocking=True)
        targets_gpu = (
            [t.to(ctx.device, non_blocking=True) for t in targets]
            if ctx.use_precomputed else None
        )

        result = forward_model(
            ctx, blur, blur_sigmas, noise_sigmas, sharp, targets_gpu,
            max_stage=active_stage, active_stage=active_stage,
            blur_clean=blur_clean,
        )
        loss = compute_single_stage_loss(ctx, result)

        ctx.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if tc["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(ctx.all_params, tc["grad_clip"])
        ctx.optimizer.step()

        bs = blur.shape[0]
        train_loss_sum += loss.item() * bs
        train_count += bs

        if ctx.logger and step % tc["log_every"] == 0:
            ctx.logger.info(
                f"[Train] E{epoch} S{step} "
                f"stage={active_stage} "
                f"loss={loss.item():.5f}"
            )

    return train_loss_sum / max(train_count, 1)
