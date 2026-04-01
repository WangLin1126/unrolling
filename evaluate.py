#!/usr/bin/env python3
"""Test / evaluate a trained unrolled deblurring model.

Visualisation layout (2 rows):
  Row 1:  Blur | Stage_a | Stage_b | ... | Final pred | GT
  Row 2:  Error | Error   | Error   | ... | Error      | (blank)

Usage:
    python test.py --config configs/default.yaml --checkpoint results/.../train/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datasets.synth_deblur import SyntheticNonBlindDeblur, BlurConfig
from models.unrolled_net import UnrolledDeblurNet
from utils.frequency import radial_average_psd, frequency_band_error


def test_collate_fn(batch):
    blurs = [b["blur"] for b in batch]
    sharps = [b["sharp"] for b in batch]
    blur_sigmas = torch.tensor([b["blur_sigma"] for b in batch], dtype=torch.float32)
    noise_sigmas = torch.tensor([b["noise_sigma"] for b in batch], dtype=torch.float32)
    paths = [b["path"] for b in batch]

    shapes = [x.shape for x in blurs]
    need_pad = len(set(shapes)) > 1
    orig_sizes = [(s[1], s[2]) for s in shapes]

    if need_pad:
        max_h = max(s[1] for s in shapes)
        max_w = max(s[2] for s in shapes)
        def _pad(t):
            _, h, w = t.shape
            if h == max_h and w == max_w:
                return t
            return F.pad(t, (0, max_w - w, 0, max_h - h), mode="reflect")
        blurs = [_pad(b) for b in blurs]
        sharps = [_pad(s) for s in sharps]

    return {
        "blur": torch.stack(blurs),
        "sharp": torch.stack(sharps),
        "blur_sigma": blur_sigmas,
        "noise_sigma": noise_sigmas,
        "paths": paths,
        "orig_sizes": orig_sizes,
    }

# ── Logging ────────────────────────────────────────────────────────

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ── Metrics ─────────────────────────────────────────────────────────

def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def _build_ssim_window(channels: int, window_size: int, dtype, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    return (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


def calc_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11,
              ssim_window: torch.Tensor | None = None) -> float:
    C = pred.shape[0]
    if ssim_window is None:
        ssim_window = _build_ssim_window(C, window_size, pred.dtype, pred.device)

    p, t = pred.unsqueeze(0), target.unsqueeze(0)
    pad = window_size // 2
    mu_p = F.conv2d(p, ssim_window, padding=pad, groups=C)
    mu_t = F.conv2d(t, ssim_window, padding=pad, groups=C)
    sig_p2 = F.conv2d(p ** 2, ssim_window, padding=pad, groups=C) - mu_p ** 2
    sig_t2 = F.conv2d(t ** 2, ssim_window, padding=pad, groups=C) - mu_t ** 2
    sig_pt = F.conv2d(p * t, ssim_window, padding=pad, groups=C) - mu_p * mu_t

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / (
        (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p2 + sig_t2 + C2)
    )
    return ssim_map.mean().item()


# ── Visualisation ───────────────────────────────────────────────────

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def _select_display_stages(
    stage_outputs: list[torch.Tensor],
    num_vis: int,
) -> list[tuple[int, torch.Tensor]]:
    T = len(stage_outputs)
    n_stage_slots = max(1, num_vis - 1)

    if n_stage_slots >= T:
        indices = list(range(T))
    elif n_stage_slots == 1:
        indices = [T - 1]
    else:
        middle_count = n_stage_slots - 1
        if middle_count == 0:
            indices = [T - 1]
        else:
            all_pts = np.round(np.linspace(0, T - 1, middle_count + 2)).astype(int)
            indices = list(all_pts[1:-1]) + [T - 1]

    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    if T - 1 not in unique_indices:
        unique_indices.append(T - 1)

    result = []
    for idx in unique_indices:
        stage_num = T - idx
        result.append((stage_num, stage_outputs[idx]))
    return result


def save_deblur_figure(
    blur: torch.Tensor,
    stage_outputs: list[torch.Tensor],
    gt: torch.Tensor,
    save_path: str,
    num_vis_stages: int = 5,
):
    selected = _select_display_stages(stage_outputs, num_vis_stages)

    displays = [("Blurry input", blur, False)]
    for stage_num, tensor in selected:
        psnr_val = calc_psnr(tensor, gt)
        if stage_num == 1:
            title = f"Final pred\nPSNR {psnr_val:.2f}"
        else:
            title = f"Stage {stage_num}\nPSNR {psnr_val:.2f}"
        displays.append((title, tensor, False))
    displays.append(("GT", gt, True))

    n_cols = len(displays)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

    for col, (title, tensor, is_gt) in enumerate(displays):
        axes[0, col].imshow(_to_numpy(tensor))
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].axis("off")

        if is_gt:
            axes[1, col].axis("off")
        else:
            err = (tensor - gt).abs()
            err_gray = _to_numpy(err).mean(axis=2)
            axes[1, col].imshow(err_gray, cmap="hot", vmin=0, vmax=max(err_gray.max(), 1e-6))
            err_psnr = calc_psnr(tensor, gt)
            axes[1, col].set_title(f"Error (PSNR {err_psnr:.2f})", fontsize=8)
            axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CATS analysis visualisation ─────────────────────────────────────

def save_spectral_convergence(
    stage_outputs: list[torch.Tensor],
    gt: torch.Tensor,
    save_path: str,
    num_bands: int = 16,
):
    """Plot radially-averaged frequency error per stage (the 'money plot').

    Creates a heatmap: x-axis = frequency band, y-axis = stage index.
    """
    T = len(stage_outputs)
    errors = np.zeros((T, num_bands))

    for t in range(T):
        band_err = frequency_band_error(stage_outputs[t], gt, num_bands=num_bands)
        errors[t] = band_err.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(
        np.log10(errors + 1e-10), aspect="auto", cmap="viridis",
        origin="lower",
    )
    ax.set_xlabel("Frequency band (low → high)")
    ax.set_ylabel("Stage index")
    ax.set_title("Log₁₀ spectral error per stage")
    plt.colorbar(im, ax=ax, label="log₁₀(MSE)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_stage_psnr_trajectory(
    stage_outputs: list[torch.Tensor],
    gt: torch.Tensor,
    save_path: str,
):
    """Plot per-stage PSNR vs stage index."""
    T = len(stage_outputs)
    psnrs = [calc_psnr(stage_outputs[t], gt) for t in range(T)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(range(1, T + 1), psnrs, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Stage")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-stage PSNR trajectory")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_stage_specialization_heatmap(
    stage_outputs: list[torch.Tensor],
    gt: torch.Tensor,
    save_path: str,
    num_bands: int = 8,
):
    """Heatmap of per-stage, per-frequency-band PSNR improvement."""
    T = len(stage_outputs)
    improvements = np.zeros((T, num_bands))

    for t in range(T):
        curr_err = frequency_band_error(stage_outputs[t], gt, num_bands=num_bands).cpu().numpy()
        if t == 0:
            # Improvement over blurry input (no prev stage output)
            improvements[t] = curr_err
        else:
            prev_err = frequency_band_error(stage_outputs[t - 1], gt, num_bands=num_bands).cpu().numpy()
            improvements[t] = prev_err - curr_err  # positive = improvement

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(improvements, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xlabel("Frequency band (low → high)")
    ax.set_ylabel("Stage index")
    ax.set_title("Stage specialization: per-band error reduction")
    plt.colorbar(im, ax=ax, label="Error reduction (MSE)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Checkpoint helpers ──────────────────────────────────────────────

def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint_for_test(checkpoint_path: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        raw_ckpt = ckpt
    else:
        state_dict = ckpt
        raw_ckpt = {"legacy_state_dict_only": True}

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    return state_dict, raw_ckpt


# ── Core test function ──────────────────────────────────────────────

def run_evaluate(cfg: dict, checkpoint_path: str, exp_dir: str | Path) -> dict:
    mc = cfg["model"]
    dc = cfg["data"]
    tc = cfg.get("test", {})

    pad_border = dc.get("pad_border", 32)
    test_batch_size = tc.get("batch_size", 1)
    num_workers = tc.get("num_workers", 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    logger = setup_logger(exp_dir / "test.log")
    logger.info("=" * 80)
    logger.info("Starting test")
    logger.info(f"Checkpoint: {Path(checkpoint_path).resolve()}")
    logger.info(f"Result dir : {exp_dir.resolve()}")
    logger.info(f"Device     : {device}")
    logger.info(f"Batch size : {test_batch_size}")
    logger.info("Test config:\n" + yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False))

    # ── Data ────────────────────────────────────────────────────
    blur_cfg = BlurConfig(**dc["blur"])
    test_glob = dc.get("test_glob", "data/val/**/*.png")
    test_ds = SyntheticNonBlindDeblur(
        test_glob,
        blur_cfg,
        pad_border=pad_border,
        T=mc["T"],
        blur_sigma_schedule_name=mc.get("blur_sigma_schedule", "uniform"),
        blur_sigma_schedule_kwargs=mc.get("blur_sigma_schedule_kwargs", {}),
    )

    logger.info(f"[TEST] {len(test_ds)} images from '{test_glob}', pad_border={pad_border}")

    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=test_collate_fn
    )

    # ── Model ───────────────────────────────────────────────────
    model = UnrolledDeblurNet(
        T=mc["T"],
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

    state_dict, raw_ckpt = load_checkpoint_for_test(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    logger.info(f"[TEST] Loaded checkpoint: {checkpoint_path}")
    if "epoch" in raw_ckpt:
        logger.info(
            f"[TEST] Checkpoint meta: epoch={raw_ckpt.get('epoch')} "
            f"best_psnr={raw_ckpt.get('best_psnr', 'N/A')} "
            f"best_val_loss={raw_ckpt.get('best_val_loss', 'N/A')}"
        )

    # ── Evaluate ────────────────────────────────────────────────
    # num_vis = tc.get("num_vis_stages", 5)
    num_vis = 6
    save_images = tc.get("save_images", True)

    psnr_list, ssim_list, name_list = [], [], []
    blur_sigma_list, noise_sigma_list = [], []
    ssim_window = None  # lazily built and cached

    fig_executor = ThreadPoolExecutor(max_workers=4) if save_images else None
    fig_futures = []

    with torch.no_grad():
        global_idx = 0
        for batch_idx, batch in enumerate(test_loader):
            blur = batch["blur"].to(device, non_blocking=True)
            sharp = batch["sharp"].to(device, non_blocking=True)
            blur_sigma = batch["blur_sigma"].to(device=device, dtype=torch.float32, non_blocking=True)
            noise_sigma = batch["noise_sigma"].to(device=device, dtype=torch.float32, non_blocking=True)
            paths = batch["paths"]
            orig_sizes = batch["orig_sizes"]

            result = model(blur, blur_sigma, noise_sigma, x_gt=None)

            pred_batch = result["pred"]
            gt_batch = sharp
            stage_outputs_batch = result["stage_outputs"]

            batch_size_actual = pred_batch.shape[0]
            for b in range(batch_size_actual):
                h_orig, w_orig = orig_sizes[b]
                pred = pred_batch[b, :, :h_orig, :w_orig]
                gt = gt_batch[b, :, :h_orig, :w_orig]
                blur_b = blur[b, :, :h_orig, :w_orig]
                sigma_val = blur_sigma[b].item()
                noise_val = noise_sigma[b].item()
                stem = Path(paths[b]).stem

                # Build SSIM window once, reuse for all images
                if ssim_window is None:
                    ssim_window = _build_ssim_window(pred.shape[0], 11, pred.dtype, pred.device)

                p = calc_psnr(pred, gt)
                s = calc_ssim(pred, gt, ssim_window=ssim_window)
                psnr_list.append(p)
                ssim_list.append(s)
                name_list.append(stem)
                blur_sigma_list.append(sigma_val)
                noise_sigma_list.append(noise_val)
                logger.info(
                    f"[{global_idx:04d}] {stem} "
                    f"PSNR={p:.2f} dB  SSIM={s:.4f}  sigma={sigma_val:.4f}  noise={noise_val:.4f}"
                )

                if save_images:
                    # Move tensors to CPU before submitting to thread pool
                    stage_outs_cpu = [so[b, :, :h_orig, :w_orig].cpu() for so in stage_outputs_batch]
                    blur_b_cpu = blur_b.cpu()
                    gt_cpu = gt.cpu()
                    save_path = str(fig_dir / f"{stem}.png")
                    fut = fig_executor.submit(
                        save_deblur_figure,
                        blur=blur_b_cpu,
                        stage_outputs=stage_outs_cpu,
                        gt=gt_cpu,
                        save_path=save_path,
                        num_vis_stages=num_vis,
                    )
                    fig_futures.append(fut)

                    # CATS analysis: spectral convergence for first 5 images
                    if global_idx < 5:
                        cats_dir = fig_dir / "cats_analysis"
                        cats_dir.mkdir(exist_ok=True)
                        fig_futures.append(fig_executor.submit(
                            save_spectral_convergence,
                            stage_outs_cpu, gt_cpu,
                            str(cats_dir / f"{stem}_spectral.png"),
                        ))
                        fig_futures.append(fig_executor.submit(
                            save_stage_psnr_trajectory,
                            stage_outs_cpu, gt_cpu,
                            str(cats_dir / f"{stem}_psnr_trajectory.png"),
                        ))
                        fig_futures.append(fig_executor.submit(
                            save_stage_specialization_heatmap,
                            stage_outs_cpu, gt_cpu,
                            str(cats_dir / f"{stem}_specialization.png"),
                        ))

                global_idx += 1

    # Wait for all figure saves to complete
    for fut in fig_futures:
        fut.result()  # raises if save_deblur_figure failed
    if fig_executor is not None:
        fig_executor.shutdown(wait=False)

    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0

    logger.info(f"CATS analysis figures saved to: {fig_dir / 'cats_analysis'}")

    summary = {
        "num_images": len(psnr_list),
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "per_image_psnr": psnr_list,
        "per_image_ssim": ssim_list,
        "per_image_name": name_list,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "test_batch_size": test_batch_size,
    }

    with open(exp_dir / "summary.txt", "w") as f:
        f.write(f"Test images : {len(psnr_list)}\n")
        f.write(f"Batch size  : {test_batch_size}\n")
        f.write(f"Avg PSNR    : {avg_psnr:.2f} dB\n")
        f.write(f"Avg SSIM    : {avg_ssim:.4f}\n")
        f.write(f"Checkpoint  : {checkpoint_path}\n\n")
        for i, (name, p, s, bsig, nsig) in enumerate(zip(name_list, psnr_list, ssim_list, blur_sigma_list, noise_sigma_list)):
            f.write(f"[{i:04d}] {name}  PSNR={p:.2f}  SSIM={s:.4f}  blur_sigma={bsig:.4f}  noise_sigma={nsig:.4f}\n")

    with open(exp_dir / "test_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 50)
    logger.info(f"Test images : {len(psnr_list)}")
    logger.info(f"Batch size  : {test_batch_size}")
    logger.info(f"Avg PSNR    : {avg_psnr:.2f} dB")
    logger.info(f"Avg SSIM    : {avg_ssim:.4f}")
    logger.info(f"Results in  : {exp_dir.resolve()}")
    logger.info("=" * 50)

    return summary


# ── Standalone ──────────────────────────────────────────────────────

def infer_test_dir_from_checkpoint(checkpoint_path: str | Path) -> Path:
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    train_dir = ckpt_path.parent
    exp_dir = train_dir.parent
    return exp_dir / "test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Optional output dir. If omitted, defaults to sibling test/ directory of the checkpoint's train/ directory.",
    )
    parser.add_argument(
        "--prefer_ckpt_config",
        action="store_true",
        help="Prefer config stored inside checkpoint when available.",
    )
    parser.add_argument("--test.batch_size", dest="test_batch_size", type=int, default=None)
    parser.add_argument("--test.num_workers", dest="test_num_workers", type=int, default=None)
    parser.add_argument("--test.glob", dest="test_glob", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, raw_ckpt = load_checkpoint_for_test(args.checkpoint, device)

    if args.prefer_ckpt_config and isinstance(raw_ckpt, dict) and "config" in raw_ckpt:
        cfg = raw_ckpt["config"]

    # Apply CLI overrides (after prefer_ckpt_config so CLI has highest priority)
    if args.test_batch_size is not None:
        cfg.setdefault("test", {})["batch_size"] = args.test_batch_size
    if args.test_num_workers is not None:
        cfg.setdefault("test", {})["num_workers"] = args.test_num_workers
    if args.test_glob is not None:
        cfg.setdefault("data", {})["test_glob"] = args.test_glob
    exp_dir = Path(args.exp_dir) if args.exp_dir is not None else infer_test_dir_from_checkpoint(args.checkpoint)
    run_evaluate(cfg, args.checkpoint, exp_dir)


if __name__ == "__main__":
    main()