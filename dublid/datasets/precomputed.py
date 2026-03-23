"""Pre-computed blind deblurring dataset — loads pre-blurred image triplets.

Expects directory structure:
    data_dir/
        blurred/   *.png
        sharp/     *.png
        kernel/    *.png   (optional, may be empty for test-only)

Adapted from reference DUBLID BlurredImageDataset with modernized image loading.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _list_sorted_images(d: str) -> list[str]:
    suffixes = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    files = []
    for s in suffixes:
        files += glob.glob(os.path.join(d, f'*.{s}'))
    return sorted(files)


class PrecomputedBlindDeblur(Dataset):
    """Load pre-existing blurred/sharp/kernel triplets.

    Args:
        data_dir:       root directory containing blurred/, sharp/, kernel/ subdirs
        image_channels: 1 for grayscale, 3 for RGB
        kernel_size:    (Hk, Wk) bounding box; kernels are center-padded to this size
    """

    def __init__(
        self,
        data_dir: str,
        image_channels: int = 1,
        kernel_size: tuple[int, int] = (45, 45),
    ):
        self.blur_files = _list_sorted_images(os.path.join(data_dir, 'blurred'))
        self.sharp_files = _list_sorted_images(os.path.join(data_dir, 'sharp'))
        kernel_dir = os.path.join(data_dir, 'kernel')
        self.kernel_files = _list_sorted_images(kernel_dir) if os.path.isdir(kernel_dir) else []

        assert len(self.blur_files) > 0, f"No blurred images found in {data_dir}/blurred"
        assert len(self.sharp_files) == len(self.blur_files), \
            f"Mismatch: {len(self.blur_files)} blurred vs {len(self.sharp_files)} sharp images"

        self.image_channels = image_channels
        self.kernel_size = kernel_size

    def __len__(self) -> int:
        return len(self.blur_files)

    def _load_image(self, path: str) -> torch.Tensor:
        """Load image, normalize to [0,1], convert channels, ensure even size."""
        if self.image_channels == 1:
            img = Image.open(path).convert('L')
        else:
            img = Image.open(path).convert('RGB')

        img_np = np.array(img).astype(np.float32) / 255.0

        # Ensure even dimensions
        h, w = img_np.shape[:2]
        h = h - h % 2
        w = w - w % 2
        if img_np.ndim == 2:
            img_np = img_np[:h, :w]
            return torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W)
        else:
            img_np = img_np[:h, :w, :]
            return torch.from_numpy(img_np.transpose(2, 0, 1))  # (C, H, W)

    def _load_kernel(self, path: str) -> torch.Tensor:
        """Load kernel, normalize, center-pad to kernel_size."""
        k = np.array(Image.open(path).convert('L')).astype(np.float32)
        k = k / (k.sum() + 1e-10)

        Hk, Wk = self.kernel_size
        kh, kw = k.shape
        if kh != Hk or kw != Wk:
            k_padded = np.zeros((Hk, Wk), dtype=np.float32)
            h_off = (Hk - kh) // 2
            w_off = (Wk - kw) // 2
            h_start = max(0, h_off)
            w_start = max(0, w_off)
            kh_use = min(kh, Hk)
            kw_use = min(kw, Wk)
            k_padded[h_start:h_start + kh_use,
                     w_start:w_start + kw_use] = k[:kh_use, :kw_use]
            k = k_padded

        # Ensure odd dimensions
        return torch.from_numpy(k)

    def __getitem__(self, idx: int) -> dict:
        blurred = self._load_image(self.blur_files[idx])
        sharp = self._load_image(self.sharp_files[idx])

        kernel = None
        if self.kernel_files:
            kernel_idx = idx if idx < len(self.kernel_files) else idx % len(self.kernel_files)
            kernel = self._load_kernel(self.kernel_files[kernel_idx])
        else:
            # No kernel available — return zeros
            kernel = torch.zeros(self.kernel_size)

        return {
            "blurred": blurred,
            "sharp": sharp,
            "kernel": kernel,
            "path": self.blur_files[idx],
        }
