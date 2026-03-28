#!/usr/bin/env python3
"""Training script for DUBLID blind image deblurring.

Follows the unrolling repo's DDP training conventions:
  - YAML config with CLI overrides
  - torchrun-based DistributedDataParallel
  - Full checkpoint save/load with RNG state
  - File-based logging + stdout on rank 0
  - PSNR-based validation with best checkpoint tracking
  - Positive-orthant projection after each optimizer step

Usage:
    # Single GPU:
    python dublid/train.py --config dublid/configs/motion_blur.yaml

    # Multi-GPU:
    torchrun --standalone --nproc_per_node=2 dublid/train.py --config dublid/configs/motion_blur.yaml

    # With CLI overrides:
    python dublid/train.py --config dublid/configs/gaussian_blur.yaml \\
        --train.lr 1e-4 --model.num_layers 8

Reference bug fixes vs original DUBLID:
  - scheduler.step() called per-epoch (not per-batch)
  - scheduler.get_last_lr() instead of deprecated scheduler.get_lr()
  - DistributedDataParallel instead of DataParallel
  - torch.utils.tensorboard instead of tensorboardX
  - Gradient clipping uses theta/lr scaling from the paper
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from dublid.models.network import DUBLIDNet
from dublid.models.network_gaussian import DUBLIDGaussianNet
from dublid.losses import build_blind_deblur_loss
from dublid.datasets.synthetic import SyntheticBlindDeblur, BlindBlurConfig
from dublid.datasets.precomputed import PrecomputedBlindDeblur


# ── Config helpers ─────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, overrides: list[str]) -> dict:
    def parse(val: str):
        s = val.strip()
        sl = s.lower()
        if sl in ("true", "false"):
            return sl == "true"
        if sl in ("null", "none"):
            return None
        try:
            if all(ch.isdigit() for ch in sl.lstrip("+-")):
                return int(s)
        except Exception:
            pass
        try:
            return float(s)
        except Exception:
            return s

    if len(overrides) % 2 != 0:
        raise ValueError("overrides must be key/value pairs")

    for i in range(0, len(overrides), 2):
        keypath = overrides[i].lstrip("-")
        keys = keypath.split(".")
        raw = overrides[i + 1]
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        leaf = keys[-1]
        old = d.get(leaf)
        if isinstance(old, bool):
            val = raw.lower() in ("true", "1", "yes")
        elif isinstance(old, int) and not isinstance(old, bool):
            val = int(raw)
        elif isinstance(old, float):
            val = float(raw)
        elif old is None:
            val = parse(raw)
        else:
            val = raw
        d[leaf] = val
    return cfg


# ── Utilities ──────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def collate_fn(batch):
    """Collate blind deblurring batches with variable-size handling."""
    blurs = [b["blurred"] for b in batch]
    sharps = [b["sharp"] for b in batch]
    kernels = [b["kernel"] for b in batch]

    shapes = [x.shape for x in blurs]
    need_pad = len(set(shapes)) > 1

    if need_pad:
        max_h = max(s[-2] for s in shapes)
        max_w = max(s[-1] for s in shapes)

        def _pad(t):
            h, w = t.shape[-2], t.shape[-1]
            if h == max_h and w == max_w:
                return t
            return nn.functional.pad(t, (0, max_w - w, 0, max_h - h), mode="reflect")

        blurs = [_pad(b) for b in blurs]
        sharps = [_pad(s) for s in sharps]

    return {
        "blurred": torch.stack(blurs),
        "sharp": torch.stack(sharps),
        "kernel": torch.stack(kernels),
    }


def train_val_split(dataset, val_ratio: float, seed: int = 42):
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio))
    return Subset(dataset, indices[n_val:]), Subset(dataset, indices[:n_val])


# ── DDP helpers ────────────────────────────────────────────────────

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_ddp = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_ddp = False
    return use_ddp, rank, world_size, local_rank, device


def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


# ── Logging ────────────────────────────────────────────────────────

def setup_logger(log_file: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"dublid_train_rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if rank == 0:
        shared = log_file.parent / "train.log"
        if shared.resolve() != log_file.resolve():
            fh2 = logging.FileHandler(shared, encoding="utf-8")
            fh2.setLevel(logging.INFO)
            fh2.setFormatter(fmt)
            logger.addHandler(fh2)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


# ── Checkpoint ─────────────────────────────────────────────────────

def get_rng_state() -> dict:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict):
    if not state:
        return
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_random" in state:
        torch.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and "cuda_random" in state:
        torch.cuda.set_rng_state_all(state["cuda_random"])


def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m


def save_checkpoint(path, model, criterion, optimizer, scheduler,
                    epoch, best_psnr, cfg, extra=None):
    raw_model = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_psnr": best_psnr,
        "config": cfg,
        "rng_state": get_rng_state(),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(ckpt_path, model, criterion, optimizer=None,
                    scheduler=None, device="cpu", logger=None):
    if logger:
        logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_model = unwrap_model(model)

    if isinstance(ckpt, dict) and "model" in ckpt:
        raw_model.load_state_dict(ckpt["model"], strict=True)
        if "criterion" in ckpt:
            try:
                criterion.load_state_dict(ckpt["criterion"], strict=False)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load criterion: {e}")
        if optimizer and ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if "rng_state" in ckpt:
            try:
                set_rng_state(ckpt["rng_state"])
            except Exception:
                pass
        return ckpt

    raw_model.load_state_dict(ckpt, strict=True)
    return {"epoch": 0, "best_psnr": 0.0}


# ── Build model ────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    mc = cfg["model"]
    model_type = mc.get("type", "motion")
    kernel_size = tuple(mc.get("kernel_size", [45, 45]))

    common = dict(
        in_channels=cfg["data"].get("image_channels", 1),
        C=mc.get("C", 16),
        K=mc.get("K", 3),
        num_layers=mc.get("num_layers", 10),
        kernel_size=kernel_size,
        bias_init=mc.get("bias_init", 0.02),
        zeta_init=mc.get("zeta_init", 1.0),
        eta_init=mc.get("eta_init", 1.0),
        prox_scale=mc.get("prox_scale", 10.0),
        kernel_prox_init=mc.get("kernel_prox_init", 1.0),
        kernel_bias_init=mc.get("kernel_bias_init", 0.0),
        kernel_scale=mc.get("kernel_scale", 1e2),
        kernel_bias_scale=mc.get("kernel_bias_scale", 0.01),
        epsilon=mc.get("epsilon", 1e-8),
    )

    if model_type == "gaussian":
        return DUBLIDGaussianNet(
            sigma_init=mc.get("sigma_init", 2.0),
            **common,
        )
    else:
        return DUBLIDNet(**common)


# ── Build dataset ──────────────────────────────────────────────────

def build_dataset(cfg: dict):
    dc = cfg["data"]
    mc = cfg["model"]
    dataset_type = dc.get("dataset_type", "synthetic")

    if dataset_type == "precomputed":
        return PrecomputedBlindDeblur(
            data_dir=dc["test_dir"],
            image_channels=dc.get("image_channels", 1),
            kernel_size=tuple(mc.get("kernel_size", [45, 45])),
        )

    blur_cfg = BlindBlurConfig(
        blur_type=dc.get("blur_type", "motion"),
        kernel_size=tuple(mc.get("kernel_size", [45, 45])),
        patch_size=tuple(dc.get("patch_size", [256, 256])),
        image_channels=dc.get("image_channels", 1),
        angle_min=dc.get("angle_min", 0.0),
        angle_max=dc.get("angle_max", 360.0),
        length_min=dc.get("length_min", 5.0),
        length_max=dc.get("length_max", 30.0),
        kernel_dir=dc.get("train_kernel_dir", ""),
        sigma_min=dc.get("sigma_min", 0.8),
        sigma_max=dc.get("sigma_max", 4.0),
        noise_stddev=dc.get("noise_stddev", 0.01),
        noise_prob=dc.get("noise_prob", 0.0),
        grad_thr=dc.get("grad_thr", 0.05),
        thr_ratio=dc.get("thr_ratio", 0.06),
    )

    return SyntheticBlindDeblur(
        image_dir=dc["train_image_dir"],
        cfg=blur_cfg,
        kernel_dir=dc.get("train_kernel_dir", None),
    )


# ── Build experiment dir ───────────────────────────────────────────

def build_exp_dir(cfg: dict, base: str = "results") -> Path:
    mc = cfg["model"]
    dc = cfg["data"]
    model_type = mc.get("type", "motion")
    params = (
        f"dublid-{model_type}"
        f"-L{mc.get('num_layers', 10)}"
        f"-C{mc.get('C', 16)}"
        f"-k{mc.get('kernel_size', [45,45])}"
        f"-ch{dc.get('image_channels', 1)}"
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / "dublid" / params / ts


# ── Main ───────────────────────────────────────────────────────────

def main():
    use_ddp, rank, world_size, local_rank, device = setup_ddp()
    logger = None

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str,
                            default="dublid/configs/motion_blur.yaml")
        args, unknown = parser.parse_known_args()

        cfg = load_config(args.config)
        if unknown:
            cfg = override_config(cfg, unknown)

        tc = cfg["train"]
        mc = cfg["model"]
        dc = cfg["data"]

        seed_everything(tc["seed"] + rank)

        # ── Experiment directory ─────────────────────────────────
        ckpt_cfg = mc.get("checkpoint", None)
        if ckpt_cfg:
            ckpt_path = Path(ckpt_cfg).expanduser().resolve()
            if ckpt_path.is_dir():
                ckpt_path = ckpt_path / "best.pth"
            train_dir = ckpt_path.parent
        else:
            ckpt_path = None
            exp_dir = build_exp_dir(cfg)
            train_dir = exp_dir / "train"

        train_dir.mkdir(parents=True, exist_ok=True)
        rank_log = train_dir / f"train_rank{rank}.log"
        logger = setup_logger(rank_log, rank=rank)

        if is_main_process():
            with open(train_dir / "config.yaml", "w") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

        logger.info("=" * 70)
        logger.info("DUBLID blind deblurring training")
        logger.info(f"Rank={rank} | World={world_size} | Device={device}")
        logger.info(f"Train dir: {train_dir.resolve()}")
        if is_main_process():
            logger.info("Config:\n" + yaml.safe_dump(cfg, default_flow_style=False))

        # ── Data ─────────────────────────────────────────────────
        full_ds = build_dataset(cfg)
        val_ratio = dc.get("val_ratio", 0.1)
        train_ds, val_ds = train_val_split(full_ds, val_ratio, seed=tc["seed"])

        train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

        train_loader = DataLoader(
            train_ds,
            batch_size=dc.get("batch_size", 16),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=dc.get("num_workers", 12),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=dc.get("num_workers", 12) > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=dc.get("batch_size", 16),
            shuffle=False,
            sampler=val_sampler,
            num_workers=dc.get("num_workers", 12),
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=dc.get("num_workers", 12) > 0,
        )

        if is_main_process():
            logger.info(f"Data: total={len(full_ds)} train={len(train_ds)} val={len(val_ds)}")

        # ── Model ────────────────────────────────────────────────
        model = build_model(cfg).to(device)
        criterion = build_blind_deblur_loss(cfg).to(device)

        if use_ddp:
            model = DDP(model, device_ids=[local_rank],
                        output_device=local_rank, broadcast_buffers=False)

        n_params = sum(p.numel() for p in unwrap_model(model).parameters()
                       if p.requires_grad)
        if is_main_process():
            logger.info(f"Model type: {mc.get('type', 'motion')} | "
                        f"Layers: {mc.get('num_layers', 10)} | "
                        f"Params: {n_params:,}")

        # ── Optimizer / Scheduler ────────────────────────────────
        optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"])

        sched_type = tc.get("scheduler", "step")
        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=tc["epochs"])
        elif sched_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=tc.get("step_size", 50),
                gamma=tc.get("gamma", 0.5),
            )
        else:
            scheduler = None

        # ── Resume ───────────────────────────────────────────────
        start_epoch = 1
        best_psnr = 0.0

        if ckpt_path and ckpt_path.exists():
            ckpt = load_checkpoint(
                ckpt_path, model, criterion, optimizer, scheduler,
                device=device, logger=logger,
            )
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_psnr = float(ckpt.get("best_psnr", 0.0))
            logger.info(f"Resumed from epoch {start_epoch}, best_psnr={best_psnr:.2f}")

        # ── Training loop ────────────────────────────────────────
        theta = tc.get("grad_clip_theta", 0.001)

        for epoch in range(start_epoch, tc["epochs"] + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            t0 = time.time()
            train_loss_sum = 0.0
            train_count = 0

            for step, batch in enumerate(train_loader, 1):
                blurred = batch["blurred"].to(device, non_blocking=True)
                sharp = batch["sharp"].to(device, non_blocking=True)
                kernel_gt = batch["kernel"].to(device, non_blocking=True)

                # Forward
                image_pred, kernel_pred = model(blurred)

                # Loss
                loss, info = criterion(
                    image_pred, sharp, kernel_pred, kernel_gt,
                    weight_list=unwrap_model(model).weight_list,
                )

                # Backward
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping (theta / lr scaling from paper)
                current_lr = optimizer.param_groups[0]["lr"]
                max_grad = theta / current_lr
                nn.utils.clip_grad_value_(model.parameters(), max_grad)

                optimizer.step()

                # Project constrained parameters to non-negative
                with torch.no_grad():
                    unwrap_model(model).project_params()

                bs = blurred.shape[0]
                train_loss_sum += loss.item() * bs
                train_count += bs

                if is_main_process() and step % tc.get("log_every", 20) == 0:
                    logger.info(
                        f"[Train] E{epoch} S{step} "
                        f"loss={loss.item():.5f} "
                        f"img={info['image_loss']:.5f} "
                        f"ker={info['kernel_loss']:.5f}"
                    )

            # scheduler.step() per-epoch (fix from reference per-batch bug)
            if scheduler is not None:
                scheduler.step()

            # Aggregate across ranks
            loss_t = torch.tensor(train_loss_sum, device=device, dtype=torch.float64)
            count_t = torch.tensor(train_count, device=device, dtype=torch.float64)
            if use_ddp:
                dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_t, op=dist.ReduceOp.SUM)

            avg_loss = (loss_t / count_t.clamp_min(1)).item()
            elapsed = time.time() - t0

            if is_main_process():
                logger.info(
                    f"[Epoch] {epoch}/{tc['epochs']} "
                    f"loss={avg_loss:.5f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"time={elapsed:.1f}s"
                )
                save_checkpoint(
                    train_dir / "last.pth",
                    model, criterion, optimizer, scheduler,
                    epoch, best_psnr, cfg,
                )

            # ── Validation ───────────────────────────────────────
            if epoch % tc.get("val_every", 1) == 0:
                model.eval()
                val_psnr_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    for batch in val_loader:
                        blurred = batch["blurred"].to(device, non_blocking=True)
                        sharp = batch["sharp"].to(device, non_blocking=True)

                        image_pred, _ = model(blurred)
                        pred = image_pred.clamp(0, 1)

                        for i in range(pred.shape[0]):
                            val_psnr_sum += psnr(pred[i], sharp[i])
                            val_count += 1

                psnr_t = torch.tensor(val_psnr_sum, device=device, dtype=torch.float64)
                cnt_t = torch.tensor(val_count, device=device, dtype=torch.float64)
                if use_ddp:
                    dist.all_reduce(psnr_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)

                avg_psnr = (psnr_t / cnt_t.clamp_min(1)).item()

                if is_main_process():
                    logger.info(f"[Val] epoch={epoch} PSNR={avg_psnr:.2f} dB")

                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        save_checkpoint(
                            train_dir / "best.pth",
                            model, criterion, optimizer, scheduler,
                            epoch, best_psnr, cfg,
                        )
                        logger.info(f"New best: {best_psnr:.2f} dB → best.pth")

                    if epoch % tc.get("save_every", 10) == 0:
                        save_checkpoint(
                            train_dir / f"ckpt_e{epoch}.pth",
                            model, criterion, optimizer, scheduler,
                            epoch, best_psnr, cfg,
                        )

        if is_main_process():
            logger.info(f"Training done. Best PSNR: {best_psnr:.2f} dB")

    except Exception as e:
        if logger:
            logger.exception("Training failed.")
        else:
            traceback.print_exc()
        raise
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
