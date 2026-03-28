"""Gradual-in-epoch: T separate forward-backward passes per batch."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import TrainContext, forward_model, unwrap_model


def train_one_epoch_gradual_in_epoch(
    ctx: TrainContext,
    train_loader,
    epoch: int,
) -> float:
    tc = ctx.cfg["train"]
    T = ctx.T
    train_loss_sum = 0.0
    train_count = 0

    assert ctx.optimizers is not None, "gradual_in_epoch requires per-stage optimizers"
    raw_model = unwrap_model(ctx.model)
    raw_crit = unwrap_model(ctx.criterion)

    for step, (blur, sharp, blur_sigmas, noise_sigmas, targets) in enumerate(train_loader, 1):
        blur = blur.to(ctx.device, non_blocking=True)
        sharp = sharp.to(ctx.device, non_blocking=True)
        blur_sigmas = blur_sigmas.to(ctx.device, non_blocking=True)
        noise_sigmas = noise_sigmas.to(ctx.device, non_blocking=True)
        targets_gpu = (
            [t.to(ctx.device, non_blocking=True) for t in targets]
            if ctx.use_precomputed else None
        )

        per_stage_losses: list[float] = []

        for t_stage in range(T):
            ctx.optimizers[t_stage].zero_grad(set_to_none=True)

            result = forward_model(
                ctx, blur, blur_sigmas, noise_sigmas, sharp, targets_gpu,
                max_stage=t_stage, active_stage=t_stage,
            )
            loss_t = raw_crit.base_loss(
                result["stage_outputs"][-1],
                result["stage_targets"][-1],
            )
            loss_t.backward()

            if tc["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(
                    list(raw_model.denoisers[t_stage].parameters()),
                    tc["grad_clip"],
                )
            ctx.optimizers[t_stage].step()
            per_stage_losses.append(loss_t.item())

        bs = blur.shape[0]
        total_loss_val = sum(per_stage_losses)
        train_loss_sum += total_loss_val * bs
        train_count += bs

        if ctx.logger and step % tc["log_every"] == 0:
            ctx.logger.info(
                f"[Train] E{epoch} S{step} "
                f"total_loss={total_loss_val:.5f} "
                f"stage_losses={[f'{l:.4f}' for l in per_stage_losses]}"
            )

    return train_loss_sum / max(train_count, 1)
