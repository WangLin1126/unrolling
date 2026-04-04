"""Synthetic non-blind deblurring dataset — full-image, precomputed targets.

Pipeline per image:
    1. Load full image x  (H, W)
    2. Reflect-pad → FFT blur with total sigma → crop → y (artifact-free)
    3. Compute T intermediate targets via blur sigma decomposition:
       targets[0] = x_gt (crop of padded),  targets[t] = blur(targets[t-1], δ_t)
       Each target is cropped to (H, W) immediately.
    4. Return blur, sharp, blur_sigma, noise_sigma, and list of T+1 targets on CPU

Targets are precomputed on CPU so the model forward() never needs
_compute_targets on GPU — saving both time and memory.

Note: if schedule is 'trainable', deltas change each step so the model
must recompute targets on GPU. The dataset targets are only used when
schedule is 'uniform'.
"""

import glob
import math
import random
from typing import Union
from dataclasses import dataclass
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from models.fft_ops import build_blur_operator, fft_conv2d_circular, precompute_freq_sq
from models.schedule import build_blur_sigma_schedule

def _parse_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


@dataclass
class BlurConfig:
    blur_type: str = "gauss"
    sigma_min: float = 0.8
    sigma_max: float = 3.0
    sigma_list: str = ""
    kernel_size: int = -1           # -1 = analytic (infinite), >0 = truncated Gaussian
    noise_prob: float = 0.5
    noise_sigma_min: float = 0.0
    noise_sigma_max: float = 0.01


class SyntheticNonBlindDeblur(Dataset):
    """
    Args:
        image_glob:  glob pattern for images
        cfg:         BlurConfig
        pad_border:  reflect-pad pixels before FFT (same as model)
        T:           number of unrolling stages (for precomputing targets)
        blur_sigma_schedule_name: schedule type for blur sigma decomposition
        blur_sigma_schedule_kwargs: kwargs for the schedule
    """

    def __init__(
        self,
        image_glob: str,
        cfg: BlurConfig = BlurConfig(),
        pad_border: int = 32,
        T: int = 5,
        blur_sigma_schedule_name: str = "uniform",
        blur_sigma_schedule_kwargs: dict | None = None,
        ):
        self.paths = sorted(glob.glob(image_glob))
        assert len(self.paths) > 0, f"No images found for {image_glob}"
        self.cfg = cfg
        self.pad_border = pad_border
        self.T = T
        self._sigma_candidates = _parse_list(cfg.sigma_list) if cfg.sigma_list else None
        self.blur_sigma_schedule = build_blur_sigma_schedule(
            blur_sigma_schedule_name, T=T, **(blur_sigma_schedule_kwargs or {}),
        )

    def __len__(self):
        return len(self.paths)

    def _sample_sigma(self):
        if self._sigma_candidates:
            return random.choice(self._sigma_candidates)
        return random.uniform(self.cfg.sigma_min, self.cfg.sigma_max)

    @torch.no_grad()
    def _compute_targets_cpu(self, x_pad, blur_sigma, H, W, p):
        """Compute T+1 intermediate targets on CPU, cropped to (H, W).

        Returns:
            list of T+1 tensors, each (C, H, W):
                targets[0] = x_gt (clean)
                targets[t] = g_{δ_t} * targets[t-1]  (progressively blurred)
        """
        Hp, Wp = x_pad.shape[-2:]
        sigma_tensor = torch.tensor(blur_sigma, device=x_pad.device, dtype=x_pad.dtype)
        blur_sigma_deltas = self.blur_sigma_schedule(sigma_tensor)  # (T,)
        freq_sq = precompute_freq_sq(Hp, Wp, x_pad.device, x_pad.dtype)

        targets = [x_pad[:, :, p:p+H, p:p+W].squeeze(0)]  # targets[0] = clean
        current = x_pad
        for t in range(self.T):
            otf_t = build_blur_operator(
                blur_sigma_deltas[t], Hp, Wp,
                kernel_size=self.cfg.kernel_size,
                device=current.device, dtype=current.dtype,
                freq_sq=freq_sq,
            ).otf
            current = fft_conv2d_circular(current, otf_t)
            targets.append(current[:, :, p:p+H, p:p+W].squeeze(0))
        return targets  # list of T+1 tensors, each (C, H, W). targets[0] = clear image.

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        # ensure even dimensions
        w, h = img.size
        w = w - w % 2
        h = h - h % 2
        img = img.crop((0, 0, w, h))

        x = TF.to_tensor(img).unsqueeze(0)  # (1, C, H, W)
        _, C, H, W = x.shape
        p = self.pad_border

        blur_sigma = self._sample_sigma()

        # ── Reflect-pad → blur → crop ──
        x_pad = F.pad(x, (p, p, p, p), mode="reflect")
        Hp, Wp = H + 2 * p, W + 2 * p

        otf = build_blur_operator(
            blur_sigma, Hp, Wp,
            kernel_size=self.cfg.kernel_size,
            device=x.device, dtype=x.dtype,
        ).otf
        y_pad = fft_conv2d_circular(x_pad, otf)
        y = y_pad[:, :, p:p+H, p:p+W]

        # add noise
        noise_sigma = 0.0
        if random.random() < self.cfg.noise_prob:
            noise_sigma = random.uniform(self.cfg.noise_sigma_min, self.cfg.noise_sigma_max)
            y = y + noise_sigma * torch.randn_like(y)
        y = y.clamp(0, 1)

        # ── Precompute intermediate targets on CPU ──
        targets = self._compute_targets_cpu(x_pad, blur_sigma, H, W, p)

        return {
            "blur": y.squeeze(0),             # (C, H, W)
            "sharp": x.squeeze(0),            # (C, H, W)
            "blur_sigma": blur_sigma,
            "noise_sigma": noise_sigma,
            "targets": targets,               # list of T+1 tensors, each (C, H, W)
            "path": self.paths[idx],
        }
