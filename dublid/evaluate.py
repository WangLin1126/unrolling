#!/usr/bin/env python3
"""Evaluation / testing script for DUBLID blind image deblurring.

Follows the unrolling repo's evaluation conventions:
  - PSNR / SSIM metrics
  - Save deblurred images and estimated kernels
  - JSON summary output
  - Kernel post-processing (normalize, remove small connected components)

Usage:
    python dublid/evaluate.py --config dublid/configs/motion_blur.yaml \\
        --checkpoint results/dublid/.../train/best.pth \\
        --output_dir results/dublid/.../test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dublid.models.network import DUBLIDNet
from dublid.models.network_gaussian import DUBLIDGaussianNet
from dublid.datasets.precomputed import PrecomputedBlindDeblur
from dublid.datasets.synthetic import SyntheticBlindDeblur, BlindBlurConfig
from dublid.train import load_config, override_config, build_model, build_dataset


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def ssim_channel(x: np.ndarray, y: np.ndarray,
                 C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """SSIM for single-channel images."""
    from scipy.ndimage import uniform_filter
    mu_x = uniform_filter(x, size=11)
    mu_y = uniform_filter(y, size=11)
    sigma_x2 = uniform_filter(x ** 2, size=11) - mu_x ** 2
    sigma_y2 = uniform_filter(y ** 2, size=11) - mu_y ** 2
    sigma_xy = uniform_filter(x * y, size=11) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / den))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.ndim == 2:
        return ssim_channel(pred, target)
    return float(np.mean([
        ssim_channel(pred[:, :, c], target[:, :, c])
        for c in range(pred.shape[2])
    ]))


def remove_small_objects_np(k: np.ndarray, min_size: int = 8) -> np.ndarray:
    """Remove small connected components from kernel."""
    try:
        from skimage.morphology import remove_small_objects
        mask = remove_small_objects(k > 0, min_size=min_size)
        return k * mask
    except ImportError:
        # Fallback: just threshold small values
        threshold = k.max() * 0.01
        k[k < threshold] = 0
        return k


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) tensor to (H, W, C) or (H, W) numpy array."""
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
    return arr


def save_image(arr: np.ndarray, path: str):
    """Save numpy array as image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def run_evaluate(cfg: dict, checkpoint_path: str, output_dir: str):
    """Run evaluation on test dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mc = cfg["model"]
    dc = cfg["data"]
    test_cfg = cfg.get("test", {})

    # Build model and load checkpoint
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()

    # Build test dataset
    test_dir = dc.get("test_dir", "")
    if test_dir and os.path.isdir(os.path.join(test_dir, "blurred")):
        dataset = PrecomputedBlindDeblur(
            data_dir=test_dir,
            image_channels=dc.get("image_channels", 1),
            kernel_size=tuple(mc.get("kernel_size", [45, 45])),
        )
    else:
        dataset = build_dataset(cfg)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Output directories
    out_dir = Path(output_dir)
    img_dir = out_dir / "images"
    ker_dir = out_dir / "kernels"
    img_dir.mkdir(parents=True, exist_ok=True)
    ker_dir.mkdir(parents=True, exist_ok=True)

    do_remove = test_cfg.get("remove_small_objects", True)
    min_obj_size = test_cfg.get("min_object_size", 8)

    results = []
    total_time = 0.0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            blurred = batch["blurred"].to(device)
            sharp = batch["sharp"].to(device)
            kernel_gt = batch["kernel"]

            t0 = time.time()
            image_pred, kernel_pred = model(blurred)
            elapsed = time.time() - t0
            total_time += elapsed

            # Convert to numpy
            pred_np = to_numpy_image(image_pred[0])
            gt_np = to_numpy_image(sharp[0])
            ker_np = kernel_pred[0].detach().cpu().numpy()

            # Post-process kernel
            if do_remove:
                ker_np = remove_small_objects_np(ker_np, min_size=min_obj_size)
            ker_sum = ker_np.sum()
            if ker_sum > 0:
                ker_np = ker_np / ker_sum

            # Metrics
            p = psnr(pred_np, gt_np)
            s = compute_ssim(pred_np, gt_np)

            results.append({
                "index": i,
                "psnr": p,
                "ssim": s,
                "time": elapsed,
                "path": batch.get("path", [""])[0] if isinstance(batch.get("path"), list) else "",
            })

            # Save images
            if test_cfg.get("save_images", True):
                fname = f"{i:05d}.png"
                save_image(pred_np, str(img_dir / fname))
                # Save kernel as normalized grayscale
                ker_vis = ker_np / (ker_np.max() + 1e-10)
                save_image(ker_vis, str(ker_dir / fname))

            print(f"[{i+1}/{len(loader)}] PSNR={p:.2f} SSIM={s:.4f} time={elapsed:.3f}s")

    # Summary
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])
    avg_time = total_time / max(len(results), 1)

    summary = {
        "num_images": len(results),
        "avg_psnr": float(avg_psnr),
        "avg_ssim": float(avg_ssim),
        "avg_time": float(avg_time),
        "total_time": float(total_time),
        "per_image": results,
    }

    summary_path = out_dir / "results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} "
          f"avg_time={avg_time:.3f}s")
    print(f"Saved to: {out_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="dublid/configs/motion_blur.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/dublid/test")
    args, unknown = parser.parse_known_args()

    cfg = load_config(args.config)
    if unknown:
        cfg = override_config(cfg, unknown)

    run_evaluate(cfg, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
