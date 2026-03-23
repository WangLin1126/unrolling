"""Synthetic blind deblurring dataset — on-the-fly blur generation.

Supports two blur types:
  - "motion": random motion kernels via angle/length parameterization
  - "gaussian": random isotropic Gaussian kernels via sigma parameterization

Pipeline per image:
  1. Load image, random crop to (Hv + Hk - 1, Wv + Wk - 1)
  2. Validate patch has sufficient edges (reject over-smooth patches)
  3. Select/generate kernel (motion or Gaussian)
  4. Convolve to create blurred observation
  5. Optionally add noise
  6. Return blurred, sharp (center crop), kernel
"""

from __future__ import annotations

import glob
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from utils.kernels import gaussian_kernel2d, motion_kernel2d


@dataclass
class BlindBlurConfig:
    """Configuration for synthetic blind deblurring data."""
    blur_type: str = "motion"          # "motion" | "gaussian"
    kernel_size: tuple[int, int] = (45, 45)
    patch_size: tuple[int, int] = (256, 256)
    image_channels: int = 1            # 1=grayscale, 3=RGB

    # Motion blur parameters
    angle_min: float = 0.0
    angle_max: float = 360.0
    length_min: float = 5.0
    length_max: float = 30.0
    kernel_dir: str = ""               # if set, load kernels from directory

    # Gaussian blur parameters
    sigma_min: float = 0.8
    sigma_max: float = 4.0

    # Noise
    noise_stddev: float = 0.01
    noise_prob: float = 0.0            # probability of adding noise

    # Patch validation
    grad_thr: float = 0.05
    thr_ratio: float = 0.06
    max_trial: int = 10


def _list_image_files(image_dir: str) -> list[str]:
    suffixes = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    files = []
    for s in suffixes:
        files += sorted(glob.glob(f"{image_dir}/*.{s}"))
        files += sorted(glob.glob(f"{image_dir}/**/*.{s}", recursive=True))
    return sorted(set(files))


def _compute_gradient_ratio(patch: np.ndarray, thr: float) -> float:
    """Compute fraction of pixels with gradient above threshold."""
    gray = patch if patch.ndim == 2 else np.mean(patch, axis=-1)
    # Simple Sobel-like gradient approximation
    dy = np.abs(gray[1:, :] - gray[:-1, :])
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    grad_mag = np.zeros_like(gray)
    grad_mag[:-1, :] += dy
    grad_mag[:, :-1] += dx
    return float(np.count_nonzero(grad_mag > thr) / grad_mag.size)


class SyntheticBlindDeblur(Dataset):
    """On-the-fly synthetic blind deblurring dataset.

    Args:
        image_dir:  directory containing clean training images
        cfg:        BlindBlurConfig
        kernel_dir: optional directory with pre-existing motion kernel images.
                    If provided and cfg.blur_type == "motion", kernels are loaded
                    from here instead of generated synthetically.
    """

    def __init__(
        self,
        image_dir: str,
        cfg: BlindBlurConfig = BlindBlurConfig(),
        kernel_dir: str | None = None,
    ):
        self.image_files = _list_image_files(image_dir)
        assert len(self.image_files) > 0, f"No images found in {image_dir}"
        self.cfg = cfg

        # Load external kernels if available
        self.external_kernels: list[np.ndarray] | None = None
        kdir = kernel_dir or cfg.kernel_dir
        if kdir and cfg.blur_type == "motion":
            kfiles = _list_image_files(kdir)
            if kfiles:
                self.external_kernels = []
                for kf in kfiles:
                    k = np.array(Image.open(kf).convert('L')).astype(np.float32)
                    k = k / (k.sum() + 1e-10)
                    self.external_kernels.append(k)

    def __len__(self) -> int:
        return len(self.image_files)

    def _generate_kernel(self) -> torch.Tensor:
        """Generate or sample a blur kernel."""
        cfg = self.cfg
        Hk, Wk = cfg.kernel_size

        if self.external_kernels is not None:
            # Sample from external kernels
            k_np = random.choice(self.external_kernels)
            k = torch.from_numpy(k_np).float()
            # Pad/crop to kernel_size
            kh, kw = k.shape
            if kh != Hk or kw != Wk:
                k_padded = torch.zeros(Hk, Wk)
                h_off = (Hk - kh) // 2
                w_off = (Wk - kw) // 2
                h_start = max(0, h_off)
                w_start = max(0, w_off)
                kh_use = min(kh, Hk)
                kw_use = min(kw, Wk)
                k_padded[h_start:h_start + kh_use,
                         w_start:w_start + kw_use] = k[:kh_use, :kw_use]
                k = k_padded
            k = k / (k.sum() + 1e-10)
            return k

        if cfg.blur_type == "motion":
            angle = random.uniform(cfg.angle_min, cfg.angle_max)
            length = random.uniform(cfg.length_min, cfg.length_max)
            k = motion_kernel2d(max(Hk, Wk), angle, length)
            # Pad to kernel_size if needed
            kh, kw = k.shape
            if kh != Hk or kw != Wk:
                k_padded = torch.zeros(Hk, Wk)
                h_off = (Hk - kh) // 2
                w_off = (Wk - kw) // 2
                k_padded[h_off:h_off + kh, w_off:w_off + kw] = k
                k = k_padded
            return k

        elif cfg.blur_type == "gaussian":
            sigma = random.uniform(cfg.sigma_min, cfg.sigma_max)
            ks = min(Hk, Wk)
            if ks % 2 == 0:
                ks -= 1
            k = gaussian_kernel2d(ks, sigma)
            # Center-pad to kernel_size
            kh, kw = k.shape
            k_padded = torch.zeros(Hk, Wk)
            h_off = (Hk - kh) // 2
            w_off = (Wk - kw) // 2
            k_padded[h_off:h_off + kh, w_off:w_off + kw] = k
            return k_padded / (k_padded.sum() + 1e-10)

        else:
            raise ValueError(f"Unknown blur_type: {cfg.blur_type}")

    def __getitem__(self, idx: int) -> dict:
        cfg = self.cfg
        Hk, Wk = cfg.kernel_size
        Hv, Wv = cfg.patch_size
        Hp, Wp = Hv + Hk - 1, Wv + Wk - 1  # 'same' conv output size

        # Load image
        img = Image.open(self.image_files[idx]).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0

        if cfg.image_channels == 1:
            img_np = np.mean(img_np, axis=-1, keepdims=True)

        # Random augmentation: flip + rotation
        if random.random() > 0.5:
            img_np = np.fliplr(img_np).copy()
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            img_np = np.rot90(img_np, k=rot_k).copy()

        Hi, Wi = img_np.shape[0], img_np.shape[1]

        # Random crop with edge validation
        patch = None
        for _ in range(cfg.max_trial):
            if Hi < Hp or Wi < Wp:
                # Image too small — use full image, center-pad
                pad_h = max(0, Hp - Hi)
                pad_w = max(0, Wp - Wi)
                img_padded = np.pad(
                    img_np,
                    ((pad_h // 2, pad_h - pad_h // 2),
                     (pad_w // 2, pad_w - pad_w // 2),
                     (0, 0)),
                    mode='reflect',
                )
                patch = img_padded[:Hp, :Wp]
                break

            h0 = random.randint(0, Hi - Hp)
            w0 = random.randint(0, Wi - Wp)
            candidate = img_np[h0:h0 + Hp, w0:w0 + Wp]

            ratio = _compute_gradient_ratio(candidate, cfg.grad_thr)
            if ratio > cfg.thr_ratio:
                patch = candidate
                break

        if patch is None:
            # Fallback: use last candidate
            h0 = random.randint(0, max(0, Hi - Hp))
            w0 = random.randint(0, max(0, Wi - Wp))
            patch = img_np[h0:h0 + Hp, w0:w0 + Wp]

        # Generate kernel
        kernel = self._generate_kernel()  # (Hk, Wk)

        # Convolve: valid convolution → (Hv, Wv) output
        # Convert to tensor for convolution
        # patch: (Hp, Wp, C) → (1, C, Hp, Wp)
        C = patch.shape[2] if patch.ndim == 3 else 1
        if patch.ndim == 2:
            patch = patch[:, :, np.newaxis]
        patch_t = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0)

        # Perform convolution per channel
        kernel_t = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, Hk, Wk)
        blurred_channels = []
        for c in range(C):
            bc = torch.nn.functional.conv2d(
                patch_t[:, c:c+1], kernel_t, padding=0)  # valid conv
            blurred_channels.append(bc)
        blurred = torch.cat(blurred_channels, dim=1).squeeze(0)  # (C, Hv, Wv)

        # Add noise
        if cfg.noise_prob > 0 and random.random() < cfg.noise_prob:
            blurred = blurred + cfg.noise_stddev * torch.randn_like(blurred)

        # Ground truth sharp image: center crop of patch
        sharp = patch_t.squeeze(0)[
            :, Hk // 2:Hk // 2 + Hv, Wk // 2:Wk // 2 + Wv
        ]

        return {
            "blurred": blurred,       # (C, Hv, Wv)
            "sharp": sharp,           # (C, Hv, Wv)
            "kernel": kernel,         # (Hk, Wk)
            "path": self.image_files[idx],
        }
