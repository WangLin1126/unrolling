"""Tail-align: post-training fine-tuning phase with small LR.

After the main training loop finishes, run an additional ``tail_epochs``
epochs (default: total_epochs // 10) with all denoisers unfrozen and a
reduced learning rate to align the distributions across stages.
"""

from __future__ import annotations

import time
import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .common import (
    TrainContext,
    forward_model,
    compute_criterion_loss,
    unfreeze_all_denoisers,
    unwrap_model,
)

logger = logging.getLogger(__name__)


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def run_tail_align(
    ctx: TrainContext,
    train_loader,
    val_loader,
    train_sampler,
    save_checkpoint_fn,
    is_main: bool,
) -> None:
    """Run the tail-align fine-tuning phase.

    Unfreezes all denoisers, creates a fresh optimiser with a small LR
    (``tail_align_lr``, default 1/10 of the original LR), and trains for
    ``total_epochs // 10`` additional epochs with cosine annealing.
    """
    tc = ctx.cfg["train"]
    total_epochs = tc["epochs"]
    tail_epochs = max(total_epochs // 10, 1)
    tail_lr = tc.get("tail_align_lr") or tc["lr"] * 0.1

    if is_main:
        ctx.logger.info(
            f"[TailAlign] Starting tail-align phase: "
            f"{tail_epochs} epochs, lr={tail_lr:.2e}"
        )

    # Unfreeze everything
    raw_model = unwrap_model(ctx.model)
    unfreeze_all_denoisers(raw_model)

    # Fresh optimiser + cosine scheduler for the tail phase
    all_params = [p for p in ctx.model.parameters() if p.requires_grad] + \
                 [p for p in ctx.criterion.parameters() if p.requires_grad]
    tail_optimizer = torch.optim.AdamW(
        all_params, lr=tail_lr, weight_decay=tc["weight_decay"],
    )
    tail_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        tail_optimizer, T_max=tail_epochs,
    )

    for t_epoch in range(1, tail_epochs + 1):
        global_epoch = total_epochs + t_epoch

        if train_sampler is not None:
            train_sampler.set_epoch(global_epoch)

        ctx.model.train()
        ctx.criterion.train()

        t0 = time.time()
        train_loss_sum_local = 0.0
        train_count_local = 0

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

            tail_optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if tc["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(all_params, tc["grad_clip"])
            tail_optimizer.step()

            bs = blur.shape[0]
            train_loss_sum_local += loss.item() * bs
            train_count_local += bs

            if is_main and step % tc["log_every"] == 0:
                ctx.logger.info(
                    f"[TailAlign] E{t_epoch}/{tail_epochs} S{step} "
                    f"loss={loss.item():.5f}"
                )

        tail_scheduler.step()

        # ── Aggregate loss ─────────────────────────────────────
        train_loss_sum_t = torch.tensor(train_loss_sum_local, device=ctx.device, dtype=torch.float64)
        train_count_t = torch.tensor(train_count_local, device=ctx.device, dtype=torch.float64)
        if ctx.use_ddp:
            dist.all_reduce(train_loss_sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_count_t, op=dist.ReduceOp.SUM)
        avg_loss = (train_loss_sum_t / train_count_t.clamp_min(1.0)).item()
        elapsed = time.time() - t0

        if is_main:
            ctx.logger.info(
                f"[TailAlign Epoch] {t_epoch}/{tail_epochs} "
                f"loss={avg_loss:.5f} "
                f"lr={tail_optimizer.param_groups[0]['lr']:.2e} "
                f"time={elapsed:.1f}s"
            )

        # ── Validation ─────────────────────────────────────────
        if t_epoch % tc["val_every"] == 0:
            ctx.model.eval()
            ctx.criterion.eval()
            val_psnr_sum = 0.0
            val_count = 0.0

            with torch.no_grad():
                for blur, sharp, blur_sigmas, noise_sigmas, targets, blur_clean in val_loader:
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
                    pred = result["pred"]
                    for i in range(pred.shape[0]):
                        val_psnr_sum += _psnr(pred[i], sharp[i])
                        val_count += 1

            val_psnr_t = torch.tensor(val_psnr_sum, device=ctx.device, dtype=torch.float64)
            val_count_t = torch.tensor(val_count, device=ctx.device, dtype=torch.float64)
            if ctx.use_ddp:
                dist.all_reduce(val_psnr_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_count_t, op=dist.ReduceOp.SUM)
            avg_psnr = (val_psnr_t / val_count_t.clamp_min(1.0)).item()

            if is_main:
                ctx.logger.info(
                    f"[TailAlign Val] epoch={t_epoch} "
                    f"val_psnr={avg_psnr:.2f} dB"
                )
                if avg_psnr > ctx.best_psnr:
                    ctx.best_psnr = avg_psnr
                    save_checkpoint_fn(
                        ctx.train_dir / "best_tail_align.pth",
                        extra={"tag": "best_tail_align"},
                    )
                    ctx.logger.info(
                        f"Saved tail-align best: "
                        f"best_tail_align.pth (PSNR={avg_psnr:.2f})"
                    )

    # Always save final tail-align checkpoint
    if is_main:
        save_checkpoint_fn(
            ctx.train_dir / "tail_align_final.pth",
            extra={"tag": "tail_align_final"},
        )
        ctx.logger.info("Tail-align phase complete.")
