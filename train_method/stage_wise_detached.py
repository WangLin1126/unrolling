"""Stage-wise detached training: all stages run end-to-end but with
``.detach()`` between consecutive stages so that each stage's loss only
updates its own denoiser.
"""

from __future__ import annotations

import torch.nn as nn

from .common import TrainContext, forward_model, compute_criterion_loss


def train_one_epoch_stage_wise_detached(
    ctx: TrainContext,
    train_loader,
    epoch: int,
) -> float:
    """Run one epoch with detached inter-stage gradients.  Returns avg loss."""
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

        result = forward_model(
            ctx, blur, blur_sigmas, noise_sigmas, sharp, targets_gpu,
            detach_between_stages=True,
        )
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
