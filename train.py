#!/usr/bin/env python3
"""Training script for unrolled Gaussian deblurring with per-stage supervision.

Targets are precomputed by the dataset on CPU and passed to the model,
avoiding redundant GPU FFT computation each forward pass.

Multi-GPU via DataParallel with custom scatter/gather.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --train.gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from datasets.synth_deblur import SyntheticNonBlindDeblur, BlurConfig
from models.unrolled_net import UnrolledDeblurNet
from utils.losses import build_combined_loss, StagewiseLoss

# ── Helpers ─────────────────────────────────────────────────────────

def build_exp_dir(cfg: dict, base: str = "results") -> Path:
    """Build experiment directory from config.

    Returns:
        Path like results/DIV2K/T_5-solver_hqs-denoiser_dncnn-depth_8-hidden_64/20260210_143025/
    """
    mc = cfg["model"]
    dc = cfg["data"]
    tc = cfg["train"]
    dk = mc.get("denoiser_kwargs", {})

    dataset_name = dc.get("dataset_name", "DIV2K")

    # pick the relevant depth/hidden for the chosen denoiser
    denoiser = mc["denoiser"]
    if denoiser == "dncnn":
        depth_val = dk.get("depth", 8)
        hidden_val = dk.get("mid_channels", 64)
    elif denoiser == "unet_small":
        depth_val = dk.get("num_levels", 2)
        hidden_val = dk.get("base_ch", 32)
    elif denoiser == "resblock":
        depth_val = dk.get("num_blocks", 5)
        hidden_val = dk.get("mid_channels", 64)
    else:
        depth_val = "NA"
        hidden_val = "NA"

    params = (
        f"T_{mc['T']}"
        f"-solver_{mc['solver']}"
        f"-denoiser_{denoiser}"
        f"-depth_{depth_val}"
        f"-hidden_{hidden_val}"
        f"-inner_{mc.get('inner_iters', 1)}"
        f"-sigma_{mc['sigma_schedule']}"
        f"-beta_{mc.get('beta_mode','geom')}"
        f"-lossw_{'learn' if mc.get('learnable_loss_weights') else 'uniform'}"
        f"-lmode_{tc.get('loss_mode')}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return Path(base) / dataset_name / params / timestamp

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, overrides: list[str]) -> dict:
    def parse(val: str):
        s = val.strip()
        sl = s.lower()
        if sl in ("true", "false"):
            return sl == "true"
        if sl in ("null", "none"):
            return None
        # int (avoid treating 1e-3 as int)
        try:
            if sl.startswith(("0x", "-0x", "+0x")):
                return int(s, 16)
            if all(ch.isdigit() for ch in sl.lstrip("+-")):
                return int(s)
        except Exception:
            pass
        # float
        try:
            return float(s)
        except Exception:
            return s  # fallback string
    if len(overrides) % 2 != 0:
        raise ValueError("overrides must be key/value pairs, length must be even")
    for i in range(0, len(overrides), 2):
        keypath = overrides[i].lstrip("-")
        keys = keypath.split(".")
        raw = overrides[i + 1]
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})  # auto-create intermediate dicts
        leaf = keys[-1]
        old = d.get(leaf)
        if isinstance(old, bool):
            val = raw.lower() in ("true", "1", "yes")
        elif isinstance(old, int) and not isinstance(old, bool):
            val = int(raw)
        elif isinstance(old, float):
            val = float(raw)
        elif old is None:
            val = parse(raw)  # infer type when new field
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
        sigmas:  (B,) tensor
        targets: list of T+1 tensors, each (B, C, H, W)
    """
    blurs = [b["blur"] for b in batch]
    sharps = [b["sharp"] for b in batch]
    sigmas = torch.tensor([b["sigma"] for b in batch], dtype=torch.float32)
    target_lists = [b["targets"] for b in batch]  # list of lists

    T_plus_1 = len(target_lists[0])

    # check if padding needed
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

    # stack targets: list of T+1 tensors, each (B, C, H, W)
    targets_batch = [
        torch.stack([target_lists[b][t] for b in range(len(batch))])
        for t in range(T_plus_1)
    ]

    return blur_batch, sharp_batch, sigmas, targets_batch


def train_val_split(dataset, val_ratio: float, seed: int = 42):
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_ratio))
    return Subset(dataset, indices[n_val:]), Subset(dataset, indices[:n_val])


def parse_gpus(gpu_str: str) -> list[int]:
    if not gpu_str or gpu_str.lower() == "none":
        return []
    return [int(g.strip()) for g in gpu_str.split(",") if g.strip()]


# ── DataParallel wrapper ────────────────────────────────────────────

class _DPWrapper(nn.DataParallel):
    """DataParallel that handles:
      - sigma: scalar tensor replicated to each GPU
      - x_gt: always None (we use precomputed_targets instead)
      - precomputed_targets: list[Tensor] split along batch dim
    """

    def scatter(self, inputs, kwargs, device_ids):
        # inputs = (y, sigma_tensor, x_gt=None, precomputed_targets)
        y, sigma_tensor, x_gt, targets_list = inputs
        n_gpu = len(device_ids)

        y_chunks = y.chunk(n_gpu, dim=0)

        # sigma: replicate scalar to each GPU
        if sigma_tensor.dim() == 0:
            sigma_val = sigma_tensor.item()
        else:
            sigma_val = sigma_tensor[0].item()

        # targets: split each of the T+1 tensors along batch dim
        if targets_list is not None:
            targets_chunks = []  # per-GPU: list of T+1 tensors
            for i in range(n_gpu):
                targets_chunks.append([
                    t.chunk(n_gpu, dim=0)[i] for t in targets_list
                ])
        else:
            targets_chunks = [None] * n_gpu

        scattered_inputs = []
        for i in range(n_gpu):
            dev = torch.device(f"cuda:{device_ids[i]}")
            s_t = torch.tensor(sigma_val, device=dev, dtype=y.dtype)
            yi = y_chunks[i].to(dev)
            tgts = None
            if targets_chunks[i] is not None:
                tgts = [t.to(dev) for t in targets_chunks[i]]
            scattered_inputs.append((yi, s_t, None, tgts))

        scattered_kwargs = [{} for _ in range(n_gpu)]
        return scattered_inputs, scattered_kwargs

    def gather(self, outputs, output_device):
        if isinstance(outputs[0], dict):
            gathered = {}
            for key in outputs[0]:
                vals = [o[key] for o in outputs]
                if vals[0] is None:
                    gathered[key] = None
                elif isinstance(vals[0], torch.Tensor):
                    gathered[key] = torch.cat(
                        [v.to(output_device) for v in vals], dim=0
                    )
                elif isinstance(vals[0], list):
                    n_stages = len(vals[0])
                    gathered[key] = [
                        torch.cat(
                            [vals[g][s].to(output_device) for g in range(len(vals))],
                            dim=0,
                        )
                        for s in range(n_stages)
                    ]
                else:
                    gathered[key] = vals[0]
            return gathered
        return super().gather(outputs, output_device)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args, unknown = parser.parse_known_args()

    cfg = load_config(args.config)
    if unknown:
        cfg = override_config(cfg, unknown)

    tc = cfg["train"]
    mc = cfg["model"]
    dc = cfg["data"]

    seed_everything(tc["seed"])

    pad_border = dc.get("pad_border", 32)
    T = mc["T"]

    # ── Device / multi-GPU setup ────────────────────────────────
    gpu_ids = parse_gpus(tc.get("gpus", ""))
    if gpu_ids:
        device = torch.device(f"cuda:{gpu_ids[0]}")
        torch.cuda.set_device(device)
        use_dp = len(gpu_ids) > 1
        print(f"Using GPUs: {gpu_ids}  (DataParallel={use_dp})")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_ids = []
        use_dp = False
        print("Using single GPU")
    else:
        device = torch.device("cpu")
        gpu_ids = []
        use_dp = False
        print("Using CPU")

    # ── Experiment directory ────────────────────────────────────
    exp_dir = build_exp_dir(cfg)
    train_dir = exp_dir / "train"
    test_dir = exp_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)

    with open(train_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Experiment dir: {exp_dir}")

    # ── Data ────────────────────────────────────────────────────
    blur_cfg = BlurConfig(**dc["blur"])
    full_ds = SyntheticNonBlindDeblur(
        dc["train_glob"], 
        blur_cfg, 
        pad_border=pad_border, 
        T=T,
        sigma_schedule_name = mc.get("sigma_schedule"),
        sigma_schedule_kwargs = mc.get("schedule_kwargs", {}),
    )

    val_ratio = dc.get("val_ratio", 0.1)
    train_ds, val_ds = train_val_split(full_ds, val_ratio, seed=tc["seed"])
    print(f"Data: {len(full_ds)} images → train {len(train_ds)} / val {len(val_ds)}")
    print(f"Config: T={T}, pad_border={pad_border}, precomputed targets on CPU")

    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc["num_workers"], pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=tc["num_workers"], pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── Model ───────────────────────────────────────────────────
    schedule_name = mc["sigma_schedule"]
    model = UnrolledDeblurNet(
        T=T,
        solver_name=mc["solver"],
        schedule_name=schedule_name,
        denoiser_name=mc["denoiser"],
        share_denoisers=mc["share_denoisers"],
        inner_iters=mc["inner_iters"],
        in_channels=mc["in_channels"],
        pad_border=pad_border,
        denoiser_kwargs=mc.get("denoiser_kwargs", {}),
        schedule_kwargs=mc.get("schedule_kwargs", {}),
        beta_mode=mc.get("beta_mode","geom")
    ).to(device)

    if use_dp:
        model = _DPWrapper(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {mc['solver'].upper()} solver, {mc['denoiser']} denoiser, "
          f"T={T}, schedule={schedule_name}")
    print(f"Trainable params: {n_params:,}")

    # ── Loss ────────────────────────────────────────────────────
    base_loss = build_combined_loss(cfg["loss"]).to(device)
    criterion = StagewiseLoss(
        T=T,
        base_loss=base_loss,
        learnable=mc.get("learnable_loss_weights", False),
        mode=tc.get("loss_mode", "all")
    ).to(device)

    # ── Optimiser ───────────────────────────────────────────────
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

    # ── Training loop ───────────────────────────────────────────
    best_psnr = 0.0
    best_val_loss = float("inf")
    patience = tc.get("early_stop_patience", 0)
    no_improve_count = 0
    history = []
    use_precomputed = (schedule_name != "trainable")

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        criterion.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (blur, sharp, sigmas, targets) in enumerate(train_loader, 1):
            blur = blur.to(device)
            sharp = sharp.to(device)
            sigmas = sigmas.to(device)

            if use_precomputed:
                # targets already on CPU from dataloader, send to GPU
                targets_gpu = [t.to(device) for t in targets]
                result = model(blur, sigmas, None, targets_gpu)
            else:
                # trainable schedule: pass x_gt, model recomputes targets
                result = model(blur, sigmas, sharp, None)

            loss, info = criterion(result["stage_outputs"], result["stage_targets"])

            optimizer.zero_grad()
            loss.backward()
            if tc["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(all_params, tc["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()

            if step % tc["log_every"] == 0:
                w_str = ", ".join(f"{w:.3f}" for w in info["weights"])
                print(f"  [E{epoch} S{step}] loss={loss.item():.5f}  "
                      f"stage_losses={[f'{l:.4f}' for l in info['per_stage_loss']]}  "
                      f"weights=[{w_str}]")

        if scheduler is not None:
            scheduler.step()

        avg_loss = epoch_loss / max(step, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{tc['epochs']}  loss={avg_loss:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  time={elapsed:.1f}s")

        # ── Validation ──────────────────────────────────────────
        if epoch % tc["val_every"] == 0:
            model.eval()
            criterion.eval()
            val_loss_sum = 0.0
            val_psnr_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for blur, sharp, sigmas, targets in val_loader:
                    blur = blur.to(device)
                    sharp = sharp.to(device)
                    sigmas = sigmas.to(device)

                    if use_precomputed:
                        targets_gpu = [t.to(device) for t in targets]
                        result = model(blur, sigmas, None, targets_gpu)
                    else:
                        result = model(blur, sigmas, sharp, None)

                    loss_v, _ = criterion(result["stage_outputs"], result["stage_targets"])
                    val_loss_sum += loss_v.item() * blur.shape[0]

                    pred = result["pred"]
                    for i in range(pred.shape[0]):
                        val_psnr_sum += psnr(pred[i], sharp[i])
                        val_count += 1

            avg_val_loss = val_loss_sum / max(val_count, 1)
            avg_psnr = val_psnr_sum / max(val_count, 1)
            print(f"  Val PSNR: {avg_psnr:.2f} dB  Val Loss: {avg_val_loss:.5f}")

            history.append({
                "epoch": epoch, "train_loss": avg_loss,
                "val_loss": avg_val_loss, "val_psnr": avg_psnr,
            })

            raw_model = model.module if use_dp else model
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(raw_model.state_dict(), train_dir / "best.pth")
                print(f"  ✓ Saved best model (PSNR={best_psnr:.2f})")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"  ⚠ Val loss did not improve ({no_improve_count}/{patience})")

            if patience > 0 and no_improve_count >= patience:
                print(f"\n✗ Early stopping at epoch {epoch}")
                torch.save({
                    "epoch": epoch, "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_psnr": best_psnr, "early_stopped": True,
                }, train_dir / "ckpt_early_stop.pth")
                break

        if epoch % 10 == 0:
            raw_model = model.module if use_dp else model
            torch.save({
                "epoch": epoch, "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(), "best_psnr": best_psnr,
            }, train_dir / f"ckpt_e{epoch}.pth")

    with open(train_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining done. Best val PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints in: {train_dir.resolve()}")

    # ── Auto test ───────────────────────────────────────────────
    if tc.get("run_test_after_train", True):
        print(f"\n{'=' * 50}")
        print("Running test...")
        print(f"{'=' * 50}\n")

        from test import run_test
        run_test(cfg, str(train_dir / "best.pth"), str(test_dir))


if __name__ == "__main__":
    main()