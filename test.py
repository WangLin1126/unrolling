#!/usr/bin/env python3
"""Test / evaluate a trained unrolled deblurring model.

Visualisation layout (2 rows):
  Row 1:  Blur | Stage_a | Stage_b | ... | Final pred | GT
  Row 2:  Error | Error   | Error   | ... | Error      | (blank)

num_vis_stages controls how many columns (excluding GT).
E.g. T=21, num_vis_stages=5  →  Blur, Stage 6, Stage 11, Stage 16, Final  +  GT

Usage:
    python test.py --config configs/default.yaml --checkpoint results/.../train/best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.synth_deblur import SyntheticNonBlindDeblur, BlurConfig
from models.unrolled_net import UnrolledDeblurNet


# ── Metrics ─────────────────────────────────────────────────────────

def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def calc_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    C = pred.shape[0]
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    p, t = pred.unsqueeze(0), target.unsqueeze(0)
    pad = window_size // 2
    mu_p = F.conv2d(p, window, padding=pad, groups=C)
    mu_t = F.conv2d(t, window, padding=pad, groups=C)
    sig_p2 = F.conv2d(p ** 2, window, padding=pad, groups=C) - mu_p ** 2
    sig_t2 = F.conv2d(t ** 2, window, padding=pad, groups=C) - mu_t ** 2
    sig_pt = F.conv2d(p * t, window, padding=pad, groups=C) - mu_p * mu_t
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) / (
        (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p2 + sig_t2 + C2)
    )
    return ssim_map.mean().item()


# ── Visualisation ───────────────────────────────────────────────────

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) tensor → (H,W,C) numpy in [0,1]."""
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def _select_display_stages(
    stage_outputs: list[torch.Tensor],
    num_vis: int,
) -> list[tuple[int, torch.Tensor]]:
    """Pick which stage outputs to display.

    stage_outputs: [est_x[T-1], est_x[T-2], ..., est_x[0]]
        index 0     → after reversing Stage T  (least recovered)
        index T-1   → after reversing Stage 1  (final, sharpest)

    num_vis = total display slots excluding GT
            = blur(1) + intermediates + final(1)
    For T=21, num_vis=5: blur + 3 intermediates (steps 6,11,16) + final

    Returns:
        list of (stage_number_1indexed, tensor) for the stage output columns only
        (blur and GT are added separately by the caller).
        Ordered from most blurry recovery to sharpest (final).
    """
    T = len(stage_outputs)

    # how many stage output slots (= num_vis minus the blur column)
    n_stage_slots = max(1, num_vis - 1)

    if n_stage_slots >= T:
        # show all stages
        indices = list(range(T))
    elif n_stage_slots == 1:
        # only final
        indices = [T - 1]
    else:
        # evenly space (n_stage_slots) points across [0, T-1]
        # but we want: middle intermediates + final
        # Use num_vis total anchors in [0, T-1], take [1:] (drop idx 0, keep final)
        middle_count = n_stage_slots - 1  # intermediates (excluding final)
        if middle_count == 0:
            indices = [T - 1]
        else:
            # spread (middle_count + 2) even points, take middle ones + final
            all_pts = np.round(np.linspace(0, T - 1, middle_count + 2)).astype(int)
            indices = list(all_pts[1:-1]) + [T - 1]

    # deduplicate preserving order
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    # ensure final is always present
    if T - 1 not in unique_indices:
        unique_indices.append(T - 1)

    result = []
    for idx in unique_indices:
        stage_num = T - idx  # 1-indexed stage number
        result.append((stage_num, stage_outputs[idx]))

    return result


def save_deblur_figure(
    blur: torch.Tensor,
    stage_outputs: list[torch.Tensor],
    gt: torch.Tensor,
    save_path: str,
    num_vis_stages: int = 5,
):
    """Save 2-row figure.

    Row 1: Blur | selected stage outputs | Final | GT
    Row 2: Error maps for each (GT column is blank)

    Args:
        blur:             (C,H,W) blurry input
        stage_outputs:    list of T tensors [est_x[T-1], ..., est_x[0]]
        gt:               (C,H,W) ground truth
        save_path:        output file path
        num_vis_stages:   total display columns excluding GT
                          (e.g. 5 = blur + 3 intermediates + final)
    """
    selected = _select_display_stages(stage_outputs, num_vis_stages)
    # selected: [(stage_num, tensor), ...] ordered blurry→sharp

    # build display list: (title, tensor, is_gt)
    displays = []
    # blur input (always first)
    displays.append(("Blurry input", blur, False))
    # selected stages
    for stage_num, tensor in selected:
        psnr_val = calc_psnr(tensor, gt)
        if stage_num == 1:
            title = f"Final pred\nPSNR {psnr_val:.2f}"
        else:
            title = f"Stage {stage_num}\nPSNR {psnr_val:.2f}"
        displays.append((title, tensor, False))
    # GT (always last)
    displays.append(("GT", gt, True))

    n_cols = len(displays)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7))

    for col, (title, tensor, is_gt) in enumerate(displays):
        # Row 1: images
        axes[0, col].imshow(_to_numpy(tensor))
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].axis("off")

        # Row 2: error maps (blank for GT)
        if is_gt:
            axes[1, col].axis("off")
        else:
            err = (tensor - gt).abs()
            err_gray = _to_numpy(err).mean(axis=2)
            im = axes[1, col].imshow(err_gray, cmap="hot", vmin=0,
                                     vmax=max(err_gray.max(), 1e-6))
            err_psnr = calc_psnr(tensor, gt)
            axes[1, col].set_title(f"Error (PSNR {err_psnr:.2f})", fontsize=8)
            axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Core test function (importable) ────────────────────────────────

def run_test(cfg: dict, checkpoint_path: str, exp_dir: str | Path) -> dict:
    """Run full evaluation.

    Args:
        cfg:             full config dict
        checkpoint_path: path to .pth weights
        exp_dir:         directory to save test results

    Returns:
        summary dict
    """
    mc = cfg["model"]
    dc = cfg["data"]
    tc = cfg.get("test", {})

    pad_border = dc.get("pad_border", 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Data (same pad_border as model) ─────────────────────────
    blur_cfg = BlurConfig(**dc["blur"])
    test_glob = dc.get("test_glob", "data/val/**/*.png")
    test_ds = SyntheticNonBlindDeblur(test_glob, cfg=blur_cfg, pad_border=pad_border)
    print(f"[TEST] {len(test_ds)} images from '{test_glob}'  pad_border={pad_border}")

    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
    )

    # ── Model (same pad_border as dataset) ──────────────────────
    model = UnrolledDeblurNet(
        T=mc["T"],
        solver_name=mc["solver"],
        schedule_name="trainable" if mc.get("learnable_schedule") else mc["schedule"],
        denoiser_name=mc["denoiser"],
        share_denoisers=mc["share_denoisers"],
        inner_iters=mc["inner_iters"],
        in_channels=mc["in_channels"],
        pad_border=pad_border,
        denoiser_kwargs=mc.get("denoiser_kwargs", {}),
        schedule_kwargs=mc.get("schedule_kwargs", {}),
    ).to(device)

    # load weights (handle DataParallel state_dict prefix)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # strip "module." prefix from DataParallel
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[TEST] Loaded checkpoint: {checkpoint_path}")

    # ── Evaluate ────────────────────────────────────────────────
    num_vis = tc.get("num_vis_stages", 5)
    save_images = tc.get("save_images", True)
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            blur = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)
            sigma = batch["sigma"]
            if isinstance(sigma, torch.Tensor):
                sigma = sigma.item()

            # inference only — no x_gt, avoids target computation
            result = model(blur, sigma, x_gt=None)

            pred = result["pred"][0]
            gt = sharp[0]
            p = calc_psnr(pred, gt)
            s = calc_ssim(pred, gt)
            psnr_list.append(p)
            ssim_list.append(s)

            print(f"  [{i:04d}] PSNR={p:.2f} dB  SSIM={s:.4f}  σ={sigma:.2f}")

            if save_images:
                stage_outs = [so[0] for so in result["stage_outputs"]]
                save_deblur_figure(
                    blur=blur[0],
                    stage_outputs=stage_outs,
                    gt=gt,
                    save_path=str(fig_dir / f"{i:04d}.png"),
                    num_vis_stages=num_vis,
                )

    # ── Summary ─────────────────────────────────────────────────
    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))

    summary = {
        "num_images": len(psnr_list),
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "per_image_psnr": psnr_list,
        "per_image_ssim": ssim_list,
    }

    with open(exp_dir / "summary.txt", "w") as f:
        f.write(f"Test images : {len(psnr_list)}\n")
        f.write(f"Avg PSNR    : {avg_psnr:.2f} dB\n")
        f.write(f"Avg SSIM    : {avg_ssim:.4f}\n")
        f.write(f"Checkpoint  : {checkpoint_path}\n\n")
        for i, (p, s) in enumerate(zip(psnr_list, ssim_list)):
            f.write(f"  [{i:04d}] PSNR={p:.2f}  SSIM={s:.4f}\n")

    # machine-readable JSON for hyperparameter search
    import json
    with open(exp_dir / "test_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"  Test images : {len(psnr_list)}")
    print(f"  Avg PSNR    : {avg_psnr:.2f} dB")
    print(f"  Avg SSIM    : {avg_ssim:.4f}")
    print(f"  Results in  : {exp_dir.resolve()}")
    print(f"{'=' * 50}")

    return summary


# ── Standalone ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, default="results/test_standalone")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_test(cfg, args.checkpoint, args.exp_dir)


if __name__ == "__main__":
    main()