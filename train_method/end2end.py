"""End-to-end training: all stages jointly optimised (original behaviour)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import TrainContext, forward_model, compute_criterion_loss


def train_one_epoch_end2end(
    ctx: TrainContext,
    train_loader,
    epoch: int,
) -> float:
    """Run one epoch of end-to-end training.  Returns total loss sum."""
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

        result = forward_model(ctx, blur, blur_sigmas, noise_sigmas, sharp, targets_gpu, blur_clean=blur_clean)
        loss, info = compute_criterion_loss(ctx, result, sharp, blur, blur_sigmas)

        ctx.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if tc["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(ctx.all_params, tc["grad_clip"])
        ctx.optimizer.step()

        bs = blur.shape[0]
        train_loss_sum += loss.item() * bs
        train_count += bs

        if ctx.logger and step % tc["log_every"] == 0:
            w_str = ", ".join(f"{w:.3f}" for w in info["weights"])
            ctx.logger.info(
                f"[Train] E{epoch} S{step} "
                f"loss={loss.item():.5f} "
                f"stage_losses={[f'{l:.4f}' for l in info['per_stage_loss']]} "
                f"weights=[{w_str}]"
            )

    return train_loss_sum / max(train_count, 1)
