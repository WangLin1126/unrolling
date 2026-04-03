#!/usr/bin/env python3
"""Training script for unrolled Gaussian deblurring with per-stage supervision.

Features:
  - Single-node multi-GPU training via DistributedDataParallel (DDP)
  - Resume from checkpoint in cfg["model"]["checkpoint"] (directory or .pth file)
  - Full checkpoint save/load: model / criterion / optimizer / scheduler / epoch / RNG / config
  - Logging to file (and stdout on main process)
  - No history.json; all run information is written to logs

Usage:
    torchrun --standalone --nproc_per_node=2 train.py --config configs/default.yaml
    torchrun --standalone --nproc_per_node=2 train.py --config configs/default.yaml --train.lr 1e-4
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

from datasets.synth_deblur import SyntheticNonBlindDeblur, BlurConfig
from models.unrolled_net import UnrolledDeblurNet
from utils.losses import build_combined_loss, StagewiseLoss
from train_method import (
    VALID_MODES,
    TrainContext,
    build_per_stage_optimizers,
    build_per_stage_schedulers,
    train_one_epoch_end2end,
    setup_one_then_another,
    train_one_epoch_one_then_another,
    train_one_epoch_gradual_in_epoch,
    setup_gradually_freeze,
    train_one_epoch_gradually_freeze,
    train_one_epoch_stage_wise_detached,
    run_tail_align,
)


# ── Helpers ─────────────────────────────────────────────────────────

def build_exp_dir(cfg: dict, base: str = "results") -> Path:
    mc = cfg["model"]
    dc = cfg["data"]
    tc = cfg["train"]
    dataset_name = dc.get("dataset_name", "DIV2K")
    denoiser = mc["denoiser"]
    dk = mc["denoiser_kwargs"][denoiser]
    fh = "front_light-" if not mc.get("blur_sigma_schedule_kwargs", {}).get("front_heavy", True) else ""
    params = (
        f"{fh}"
        f"T{mc['T']}"
        f"-{mc['solver']}"
        f"-{denoiser}"
        f"-inner{mc.get('inner_iters', 1)}"
        f"-blur_sigma_{mc.get('blur_sigma_schedule', 'uniform')}_{dc['blur']['sigma_list']}"
        f"-noise_sigma_{dc['blur']['noise_sigma_min']}_{dc['blur']['noise_sigma_max']}"
        f"-beta_{mc.get('beta_schedule', 'geom')}"
        f"-lossw_{'learn' if mc.get('learnable_loss_weights') else 'uniform'}"
        f"-lmode_{tc.get('loss_mode', 'all')}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / dataset_name / params / timestamp

def build_cats_exp_dir(cfg: dict, base: str = "results") -> Path:
    mc = cfg["model"]
    dc = cfg["data"]
    tc = cfg["train"]
    dataset_name = dc.get("dataset_name", "DIV2K")
    denoiser = mc["denoiser"]
    df= tc['cts_kwargs']['difficulty_schedule']
    params = (
        f"{tc.get('loss_mode', 'all')}"
        f"-df-{df}"
        f"-T{mc['T']}"
        f"-{mc['solver']}"
        f"-{denoiser}"
        f"-inner{mc.get('inner_iters', 1)}"
        f"-blur_{dc['blur']['sigma_list']}"
        f"-noise_{dc['blur']['noise_sigma_min']}_{dc['blur']['noise_sigma_max']}"
        f"-beta_{mc.get('beta_schedule', 'geom')}"
        f"-filter_{tc.get('cts_kwargs').get('filter_type','gaussian')}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / dataset_name / params / timestamp


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
            if sl.startswith(("0x", "-0x", "+0x")):
                return int(s, 16)
            if all(ch.isdigit() for ch in sl.lstrip("+-")):
                return int(s)
        except Exception:
            pass
        try:
            return float(s)
        except Exception:
            return s

    if len(overrides) % 2 != 0:
        raise ValueError("overrides must be key/value pairs, length must be even")

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
    """Collate variable-size images + precomputed target lists.

    Returns:
        blur:    (B, C, H, W)
        sharp:   (B, C, H, W)
        blur_sigma:  (B,)
        noise_sigma : (B,)
        targets: list of T+1 tensors, each (B, C, H, W)
    """
    blurs = [b["blur"] for b in batch]
    sharps = [b["sharp"] for b in batch]
    blur_sigmas = torch.tensor([b["blur_sigma"] for b in batch], dtype=torch.float32)
    noise_sigmas = torch.tensor([b["noise_sigma"] for b in batch], dtype=torch.float32)
    target_lists = [b["targets"] for b in batch]

    T_plus_1 = len(target_lists[0])
    shapes = [x.shape for x in blurs]
    need_pad = len(set(shapes)) > 1

    if need_pad:
        max_h = max(s[1] for s in shapes)
        max_w = max(s[2] for s in shapes)

        def _pad(t):
            _, h, w = t.shape
            if h == max_h and w == max_w:
                return t
            return nn.functional.pad(t, (0, max_w - w, 0, max_h - h), mode="reflect")

        blurs = [_pad(b) for b in blurs]
        sharps = [_pad(s) for s in sharps]
        target_lists = [[_pad(t) for t in tl] for tl in target_lists]

    blur_batch = torch.stack(blurs)
    sharp_batch = torch.stack(sharps)
    targets_batch = [
        torch.stack([target_lists[b][t] for b in range(len(batch))])
        for t in range(T_plus_1)
    ]

    return blur_batch, sharp_batch, blur_sigmas, noise_sigmas, targets_batch


def train_val_split(dataset, val_ratio: float, seed: int = 42):
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio))
    return Subset(dataset, indices[n_val:]), Subset(dataset, indices[:n_val])


def resolve_checkpoint_path(ckpt_cfg: str | None) -> Path | None:
    if not ckpt_cfg:
        return None
    p = Path(ckpt_cfg).expanduser().resolve()
    if p.is_dir():
        p = p / "best.pth"
    return p


# ── DDP helpers ─────────────────────────────────────────────────────

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


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


# ── Logging ─────────────────────────────────────────────────────────

def setup_logger(log_file: Path, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"train_rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file.parent.mkdir(parents=True, exist_ok=True)

    # All ranks log to their own file; rank 0 also logs to shared train.log
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if rank == 0:
        shared_log = log_file.parent / "train.log"
        if shared_log.resolve() != log_file.resolve():
            fh2 = logging.FileHandler(shared_log, encoding="utf-8")
            fh2.setLevel(logging.INFO)
            fh2.setFormatter(formatter)
            logger.addHandler(fh2)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


# ── Checkpoint helpers ──────────────────────────────────────────────

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
    if isinstance(m, DDP):
        m = m.module
    # torch.compile wraps in OptimizedModule; unwrap to get original
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def save_checkpoint(
    path: Path,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_psnr: float,
    best_val_loss: float,
    no_improve_count: int,
    cfg: dict,
    extra: dict | None = None,
    optimizers: list[torch.optim.Optimizer] | None = None,
    schedulers: list | None = None,
):
    raw_model = unwrap_model(model)
    raw_criterion = unwrap_model(criterion)

    payload = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "criterion": raw_criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_psnr": best_psnr,
        "best_val_loss": best_val_loss,
        "no_improve_count": no_improve_count,
        "config": cfg,
        "rng_state": get_rng_state(),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Per-stage optimisers/schedulers for gradual_in_epoch
    if optimizers is not None:
        payload["optimizers"] = [opt.state_dict() for opt in optimizers]
    if schedulers is not None:
        payload["schedulers"] = [sch.state_dict() for sch in schedulers]

    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: torch.device | str = "cpu",
    logger: logging.Logger | None = None,
) -> dict:
    if logger:
        logger.info(f"Loading checkpoint from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    raw_model = unwrap_model(model)
    raw_criterion = unwrap_model(criterion)

    if isinstance(ckpt, dict) and "model" in ckpt:
        raw_model.load_state_dict(ckpt["model"], strict=True)

        if "criterion" in ckpt:
            try:
                raw_criterion.load_state_dict(ckpt["criterion"], strict=False)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to load criterion state: {e}")

        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])

        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])

        if "rng_state" in ckpt:
            try:
                set_rng_state(ckpt["rng_state"])
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to restore RNG state: {e}")

        return ckpt

    # legacy pure-state-dict
    raw_model.load_state_dict(ckpt, strict=True)
    return {
        "epoch": 0,
        "best_psnr": 0.0,
        "best_val_loss": float("inf"),
        "no_improve_count": 0,
        "legacy_state_dict_only": True,
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    use_ddp, rank, world_size, local_rank, device = setup_ddp()

    logger = None
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="configs/default.yaml")
        args, unknown = parser.parse_known_args()

        cfg = load_config(args.config)
        if unknown:
            cfg = override_config(cfg, unknown)

        tc = cfg["train"]
        mc = cfg["model"]
        dc = cfg["data"]

        seed_everything(tc["seed"] + rank)

        # ── GPU performance flags (FP32-safe) ──────────────────
        # cuDNN auto-tuner: benchmark convolution algorithms on first run,
        # then cache the fastest one. Free speedup when input sizes are fixed.
        torch.backends.cudnn.benchmark = True

        ckpt_cfg = mc.get("checkpoint", None)
        resume_ckpt = resolve_checkpoint_path(ckpt_cfg)

        if resume_ckpt is not None:
            if not resume_ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resume_ckpt}")
            train_dir = resume_ckpt.parent
            exp_dir = train_dir.parent
            test_dir = exp_dir / "test"
        else:
            if tc['loss_mode'].startswith("cats_"):
                exp_dir = build_cats_exp_dir(cfg)
            else:
                exp_dir = build_exp_dir(cfg)
            train_dir = exp_dir / "train"
            test_dir = exp_dir / "test"

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        rank_log = train_dir / f"train_rank{rank}.log"
        logger = setup_logger(rank_log, rank=rank)

        if is_main_process():
            with open(train_dir / "config.yaml", "w") as f:
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info(f"Rank={rank} | World size={world_size} | Local rank={local_rank}")
        logger.info(f"Device: {device}")
        logger.info(f"Experiment dir: {exp_dir.resolve()}")
        logger.info(f"Train dir: {train_dir.resolve()}")
        logger.info(f"Test dir: {test_dir.resolve()}")
        if is_main_process():
            logger.info("Full config:\n" + yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))

        pad_border = dc.get("pad_border", 32)
        T = mc["T"]

        # ── Data ────────────────────────────────────────────────
        blur_cfg = BlurConfig(**dc["blur"])
        full_ds = SyntheticNonBlindDeblur(
            dc["train_glob"],
            blur_cfg,
            pad_border=pad_border,
            T=T,
            blur_sigma_schedule_name=mc.get("blur_sigma_schedule", "uniform"),
            blur_sigma_schedule_kwargs=mc.get("blur_sigma_schedule_kwargs", {}),
        )

        val_ratio = dc.get("val_ratio", 0.1)
        train_ds, val_ds = train_val_split(full_ds, val_ratio, seed=tc["seed"])

        train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

        _nw = tc["num_workers"]
        train_loader = DataLoader(
            train_ds,
            batch_size=tc["batch_size"],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=_nw,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
            persistent_workers=_nw > 0,
            prefetch_factor=tc.get("prefetch_factor", 3) if _nw > 0 else None,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=tc["batch_size"],
            shuffle=False,
            sampler=val_sampler,
            num_workers=_nw,
            pin_memory=True,
            collate_fn=collate_fn,
            persistent_workers=_nw > 0,
            prefetch_factor=tc.get("prefetch_factor", 3) if _nw > 0 else None,
        )

        if is_main_process():
            logger.info(f"Data: total={len(full_ds)} | train={len(train_ds)} | val={len(val_ds)}")
            logger.info(f"Config summary: T={T}, pad_border={pad_border}, precomputed targets on CPU")

        # ── Model ───────────────────────────────────────────────
        model = UnrolledDeblurNet(
            T=T,
            solver_name=mc["solver"],
            blur_sigma_schedule=mc.get("blur_sigma_schedule", "uniform"),
            denoiser_name=mc["denoiser"],
            share_denoisers=mc["share_denoisers"],
            inner_iters=mc["inner_iters"],
            in_channels=mc["in_channels"],
            pad_border=pad_border,
            denoiser_kwargs=mc.get("denoiser_kwargs", {}),
            blur_sigma_schedule_kwargs=mc.get("blur_sigma_schedule_kwargs", {}),
            beta_schedule=mc.get("beta_schedule", "geom"),
            beta_kwargs=mc.get("beta_kwargs", {}),
            noise_sigma_schedule=mc.get("noise_sigma_schedule", "loguniform"),
            noise_sigma_schedule_kwargs=mc.get("noise_sigma_schedule_kwargs", {}),
        ).to(device)

        # Channels-last memory format: cuDNN prefers NHWC layout for Conv2d,
        # avoids internal transposes and enables faster kernels on H200/A100.
        if tc.get("channels_last", True):
            model = model.to(memory_format=torch.channels_last)

        if tc.get("use_compile", False):
            model = torch.compile(model)

        base_loss = build_combined_loss(cfg["loss"]).to(device)
        criterion = StagewiseLoss(
            T=T,
            base_loss=base_loss,
            learnable=mc.get("learnable_loss_weights", False),
            mode=tc.get("loss_mode", "all"),
            cts_kwargs=tc.get("cts_kwargs", None),
        ).to(device)

        # ── Stage-wise training mode ────────────────────────────
        stage_wise_mode = tc.get("stage_wise_train", "end2end")
        if stage_wise_mode not in VALID_MODES:
            raise ValueError(
                f"Unknown stage_wise_train mode '{stage_wise_mode}'. "
                f"Choose from: {sorted(VALID_MODES)}"
            )
        if stage_wise_mode != "end2end" and mc.get("share_denoisers", False):
            raise ValueError(
                "Stage-wise training is incompatible with share_denoisers=True."
            )
        if is_main_process():
            logger.info(f"Stage-wise training mode: {stage_wise_mode}")

        if use_ddp:
            ddp_kwargs = dict(
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
            if stage_wise_mode != "end2end":
                ddp_kwargs["find_unused_parameters"] = True
            model = DDP(model, **ddp_kwargs)

            criterion_has_params = any(p.requires_grad for p in criterion.parameters())
            if criterion_has_params:
                criterion = DDP(
                    criterion,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=False,
                )

        n_params = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        if is_main_process():
            logger.info(
                f"Model: {mc['solver'].upper()} solver, {mc['denoiser']} denoiser, "
                f"T={T}"
            )
            logger.info(f"Trainable params: {n_params:,}")

        # ── Optimizer / Scheduler ───────────────────────────────
        # For gradual_in_epoch we use T independent optimisers/schedulers;
        # for end2end and one_then_another a single one suffices.
        optimizers: list[torch.optim.Optimizer] | None = None
        schedulers: list | None = None

        if stage_wise_mode == "gradual_in_epoch":
            raw = unwrap_model(model)
            raw_crit = unwrap_model(criterion)
            optimizers = build_per_stage_optimizers(
                raw, raw_crit, lr=tc["lr"],
                weight_decay=tc["weight_decay"], T=T,
            )
            schedulers = build_per_stage_schedulers(
                optimizers,
                scheduler_name=tc["scheduler"],
                total_epochs=tc["epochs"],
                step_size=tc.get("step_size", 50),
                gamma=tc.get("gamma", 0.5),
            )
            # Placeholders so the rest of the code doesn't need guards
            optimizer = optimizers[0]
            scheduler = schedulers[0] if schedulers else None
            all_params = list(model.parameters()) + list(criterion.parameters())
        else:
            all_params = list(model.parameters()) + list(criterion.parameters())
            optimizer = torch.optim.AdamW(
                all_params, lr=tc["lr"], weight_decay=tc["weight_decay"]
            )

            if tc["scheduler"] == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=tc["epochs"]
                )
            elif tc["scheduler"] == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=tc["step_size"], gamma=tc["gamma"]
                )
            else:
                scheduler = None

        # ── Resume ──────────────────────────────────────────────
        start_epoch = 1
        best_psnr = 0.0
        best_val_loss = float("inf")
        patience = tc.get("early_stop_patience", 0)
        no_improve_count = 0
        use_precomputed = (mc.get("blur_sigma_schedule", "uniform") != "trainable")
        loss_mode = tc.get("loss_mode", "all")
        use_cats = loss_mode.startswith("cats_")

        if resume_ckpt is not None:
            ckpt = load_checkpoint(
                ckpt_path=resume_ckpt,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                logger=logger,
            )
            # Restore per-stage optimisers/schedulers for gradual_in_epoch
            if stage_wise_mode == "gradual_in_epoch" and "optimizers" in ckpt:
                for i, opt in enumerate(optimizers):
                    if i < len(ckpt["optimizers"]):
                        opt.load_state_dict(ckpt["optimizers"][i])
                if schedulers and "schedulers" in ckpt:
                    for i, sch in enumerate(schedulers):
                        if i < len(ckpt["schedulers"]):
                            sch.load_state_dict(ckpt["schedulers"][i])

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_psnr = float(ckpt.get("best_psnr", 0.0))
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            no_improve_count = int(ckpt.get("no_improve_count", 0))

            logger.info(
                f"Resumed training from epoch={start_epoch} "
                f"(previous epoch={ckpt.get('epoch', 0)})"
            )
            logger.info(
                f"Recovered states: best_psnr={best_psnr:.4f}, "
                f"best_val_loss={best_val_loss:.6f}, "
                f"no_improve_count={no_improve_count}"
            )
            if ckpt.get("legacy_state_dict_only", False):
                logger.warning(
                    "Loaded a legacy checkpoint containing only model.state_dict(). "
                    "Optimizer/scheduler/criterion/RNG were not restored."
                )

        # ── Build shared TrainContext ───────────────────────────
        ctx = TrainContext(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            all_params=all_params,
            device=device,
            cfg=cfg,
            T=T,
            use_precomputed=use_precomputed,
            use_cats=use_cats,
            use_ddp=use_ddp,
            train_dir=train_dir,
            logger=logger if is_main_process() else None,
            channels_last=tc.get("channels_last", True),
            optimizers=optimizers,
            schedulers=schedulers,
            best_psnr=best_psnr,
            best_val_loss=best_val_loss,
            no_improve_count=no_improve_count,
        )

        # ── Training loop ───────────────────────────────────────
        early_stop_flag = False
        prev_active_stage = -1
        prev_freeze_boundary = -2

        # Extra checkpoint kwargs for multi-optimizer modes
        _ckpt_extra_kw: dict = {}
        if stage_wise_mode == "gradual_in_epoch" and optimizers is not None:
            _ckpt_extra_kw["optimizers"] = optimizers
            _ckpt_extra_kw["schedulers"] = schedulers

        for epoch in range(start_epoch, tc["epochs"] + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            criterion.train()

            t0 = time.time()

            # ── Dispatch to strategy ───────────────────────────
            if stage_wise_mode == "end2end":
                avg_loss = train_one_epoch_end2end(ctx, train_loader, epoch)

            elif stage_wise_mode == "one_then_another":
                cur_active = setup_one_then_another(ctx, epoch)
                if cur_active != prev_active_stage:
                    if is_main_process():
                        logger.info(
                            f"[one_then_another] Epoch {epoch}: "
                            f"active denoiser stage = {cur_active}"
                        )
                    prev_active_stage = cur_active
                avg_loss = train_one_epoch_one_then_another(
                    ctx, train_loader, epoch, cur_active,
                )

            elif stage_wise_mode == "gradual_in_epoch":
                avg_loss = train_one_epoch_gradual_in_epoch(ctx, train_loader, epoch)

            elif stage_wise_mode == "gradually_freeze":
                last_frozen = setup_gradually_freeze(ctx, epoch)
                if last_frozen != prev_freeze_boundary:
                    if is_main_process():
                        desc = (
                            f"frozen denoisers 0..{last_frozen}"
                            if last_frozen >= 0 else "all trainable"
                        )
                        logger.info(
                            f"[gradually_freeze] Epoch {epoch}: {desc}"
                        )
                    prev_freeze_boundary = last_frozen
                avg_loss = train_one_epoch_gradually_freeze(
                    ctx, train_loader, epoch, last_frozen,
                )

            elif stage_wise_mode == "stage_wise_detached":
                avg_loss = train_one_epoch_stage_wise_detached(
                    ctx, train_loader, epoch,
                )

            # ── Scheduler step ─────────────────────────────────
            if stage_wise_mode == "gradual_in_epoch":
                if schedulers:
                    for sch in schedulers:
                        sch.step()
            else:
                if scheduler is not None:
                    scheduler.step()

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            if is_main_process():
                logger.info(
                    f"[Epoch] {epoch}/{tc['epochs']} "
                    f"train_loss={avg_loss:.5f} "
                    f"lr={current_lr:.2e} "
                    f"time={elapsed:.1f}s"
                )

                save_checkpoint(
                    train_dir / "last.pth",
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_psnr=best_psnr,
                    best_val_loss=best_val_loss,
                    no_improve_count=no_improve_count,
                    cfg=cfg,
                    extra={"tag": "last"},
                    **_ckpt_extra_kw,
                )

            # ── Validation ──────────────────────────────────────
            if epoch % tc["val_every"] == 0:
                model.eval()
                criterion.eval()

                val_loss_sum_local = 0.0
                val_psnr_sum_local = 0.0
                val_count_local = 0.0

                _cl = tc.get("channels_last", True)
                with torch.no_grad():
                    for blur, sharp, blur_sigmas, noise_sigmas, targets in val_loader:
                        blur = blur.to(device, non_blocking=True)
                        sharp = sharp.to(device, non_blocking=True)
                        blur_sigmas = blur_sigmas.to(device, non_blocking=True)
                        noise_sigmas = noise_sigmas.to(device, non_blocking=True)
                        if _cl:
                            blur = blur.to(memory_format=torch.channels_last)
                            sharp = sharp.to(memory_format=torch.channels_last)
                        if use_precomputed:
                            targets_gpu = [t.to(device, non_blocking=True) for t in targets]
                            if _cl:
                                targets_gpu = [t.to(memory_format=torch.channels_last) for t in targets_gpu]
                            result = model(blur=blur, blur_sigma=blur_sigmas, noise_sigma=noise_sigmas, x_gt=None, precomputed_targets=targets_gpu)
                        else:
                            result = model(blur=blur, blur_sigma=blur_sigmas, noise_sigma=noise_sigmas, x_gt=sharp, precomputed_targets=None)

                        if use_cats:
                            loss_v, _ = criterion(
                                result["stage_outputs"], result["stage_targets"],
                                x_gt=sharp, blur=blur, blur_sigma=blur_sigmas,
                            )
                        else:
                            loss_v, _ = criterion(result["stage_outputs"], result["stage_targets"])
                        val_loss_sum_local += loss_v.item() * blur.shape[0]

                        pred = result["pred"]
                        for i in range(pred.shape[0]):
                            val_psnr_sum_local += psnr(pred[i], sharp[i])
                            val_count_local += 1

                val_loss_sum_tensor = torch.tensor(val_loss_sum_local, device=device, dtype=torch.float64)
                val_psnr_sum_tensor = torch.tensor(val_psnr_sum_local, device=device, dtype=torch.float64)
                val_count_tensor = torch.tensor(val_count_local, device=device, dtype=torch.float64)

                if use_ddp:
                    dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_psnr_sum_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)

                avg_val_loss = (val_loss_sum_tensor / val_count_tensor.clamp_min(1.0)).item()
                avg_psnr = (val_psnr_sum_tensor / val_count_tensor.clamp_min(1.0)).item()

                if is_main_process():
                    logger.info(
                        f"[Val] epoch={epoch} val_psnr={avg_psnr:.2f} dB "
                        f"val_loss={avg_val_loss:.5f}"
                    )

                    if avg_psnr > best_psnr:
                        best_psnr = avg_psnr
                        save_checkpoint(
                            train_dir / "best.pth",
                            model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            best_psnr=best_psnr,
                            best_val_loss=best_val_loss,
                            no_improve_count=no_improve_count,
                            cfg=cfg,
                            extra={"tag": "best_by_psnr"},
                            **_ckpt_extra_kw,
                        )
                        logger.info(f"Saved best checkpoint: best.pth (PSNR={best_psnr:.2f})")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        no_improve_count = 0
                        logger.info("Validation loss improved.")
                    else:
                        no_improve_count += 1
                        logger.info(
                            f"Validation loss did not improve "
                            f"({no_improve_count}/{patience})"
                        )

                    if epoch % 10 == 0:
                        save_checkpoint(
                            train_dir / f"ckpt_e{epoch}.pth",
                            model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            best_psnr=best_psnr,
                            best_val_loss=best_val_loss,
                            no_improve_count=no_improve_count,
                            cfg=cfg,
                            extra={"tag": f"epoch_{epoch}"},
                            **_ckpt_extra_kw,
                        )
                        logger.info(f"Saved periodic checkpoint: ckpt_e{epoch}.pth")

                    if patience > 0 and no_improve_count >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        save_checkpoint(
                            train_dir / "ckpt_early_stop.pth",
                            model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            best_psnr=best_psnr,
                            best_val_loss=best_val_loss,
                            no_improve_count=no_improve_count,
                            cfg=cfg,
                            extra={"early_stopped": True, "tag": "early_stop"},
                            **_ckpt_extra_kw,
                        )
                        early_stop_flag = True

                if use_ddp:
                    stop_tensor = torch.tensor(1 if early_stop_flag else 0, device=device, dtype=torch.int32)
                    dist.broadcast(stop_tensor, src=0)
                    early_stop_flag = bool(stop_tensor.item())

                if early_stop_flag:
                    break

        if is_main_process():
            logger.info(f"Training done. Best val PSNR: {best_psnr:.2f} dB")
            logger.info(f"Checkpoints in: {train_dir.resolve()}")

        # ── Tail-align phase (optional) ─────────────────────────
        if tc.get("tail_align", False) and stage_wise_mode != "end2end":
            # Update ctx tracking state with latest values
            ctx.best_psnr = best_psnr
            ctx.best_val_loss = best_val_loss
            ctx.no_improve_count = no_improve_count

            def _save_ckpt_for_tail(path, extra=None):
                save_checkpoint(
                    path,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=tc["epochs"],
                    best_psnr=ctx.best_psnr,
                    best_val_loss=ctx.best_val_loss,
                    no_improve_count=ctx.no_improve_count,
                    cfg=cfg,
                    extra=extra,
                    **_ckpt_extra_kw,
                )

            run_tail_align(
                ctx=ctx,
                train_loader=train_loader,
                val_loader=val_loader,
                train_sampler=train_sampler,
                save_checkpoint_fn=_save_ckpt_for_tail,
                is_main=is_main_process(),
            )
            # Recover updated best_psnr from ctx
            best_psnr = ctx.best_psnr

            if is_main_process():
                logger.info(f"Final best PSNR (incl. tail-align): {best_psnr:.2f} dB")

        # ── Auto test ───────────────────────────────────────────
        if use_ddp:
            dist.barrier()
            
        should_run_test = tc.get("run_test_after_train", True) and is_main_process()
        # Prefer tail-align best if it exists, else fall back to main best
        best_tail = train_dir / "best_tail_align.pth"
        best_ckpt = best_tail if best_tail.exists() else train_dir / "best.pth"

        # 在测试前先退出 DDP，避免其他 rank 在 barrier/collective 中等待超时
        cleanup_ddp()

        if should_run_test:
            if best_ckpt.exists():
                logger.info("=" * 50)
                logger.info("Running test...")
                logger.info("=" * 50)

                from evaluate import run_evaluate
                run_evaluate(cfg, str(best_ckpt), str(test_dir))
            else:
                logger.warning("Skip test because best.pth does not exist.")

        return

    except Exception as e:
        if logger is None:
            # fallback logger to stderr if failure happens very early
            print("Unhandled exception before logger setup.", file=sys.stderr)
            traceback.print_exc()
        else:
            logger.exception("Unhandled exception during training.")
            try:
                # only main process saves crash checkpoint to avoid file collisions
                if is_main_process() and "train_dir" in locals():
                    _crash_kw: dict = {}
                    if locals().get("optimizers") is not None:
                        _crash_kw["optimizers"] = locals()["optimizers"]
                        _crash_kw["schedulers"] = locals().get("schedulers")
                    save_checkpoint(
                        train_dir / "crash.pth",
                        model=locals()["model"],
                        criterion=locals()["criterion"],
                        optimizer=locals()["optimizer"],
                        scheduler=locals().get("scheduler", None),
                        epoch=locals().get("epoch", 0),
                        best_psnr=locals().get("best_psnr", 0.0),
                        best_val_loss=locals().get("best_val_loss", float("inf")),
                        no_improve_count=locals().get("no_improve_count", 0),
                        cfg=locals().get("cfg", {}),
                        extra={
                            "tag": "crash",
                            "exception": repr(e),
                            "traceback": traceback.format_exc(),
                        },
                        **_crash_kw,
                    )
                    logger.info(f"Saved crash checkpoint to: {train_dir / 'crash.pth'}")
            except Exception:
                logger.exception("Failed to save crash checkpoint.")
        raise
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()