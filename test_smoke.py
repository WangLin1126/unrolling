#!/usr/bin/env python3
"""Smoke test: verify forward + loss for all solver × schedule × denoiser combos."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.unrolled_net import UnrolledDeblurNet
from utils.losses import build_combined_loss, StagewiseLoss


def test_forward(solver, schedule, denoiser, T=3, H=64, W=64):
    model = UnrolledDeblurNet(
        T=T,
        solver_name=solver,
        schedule_name=schedule,
        denoiser_name=denoiser,
        inner_iters=1,
        denoiser_kwargs={"mid_channels": 32, "depth": 4, "num_blocks": 2, "base_ch": 16},
    )
    model.eval()

    y = torch.randn(2, 3, H, W).clamp(0, 1)
    x_gt = torch.randn(2, 3, H, W).clamp(0, 1)
    sigma = 1.5

    # simulate precomputed targets (T+1 tensors on original grid)
    import math
    from models.fft_ops import gaussian_otf, fft_conv2d_circular
    delta = sigma / math.sqrt(T)
    p = 8
    import torch.nn.functional as Fn
    x_pad = Fn.pad(x_gt.unsqueeze(0) if x_gt.dim() == 3 else x_gt, (p,p,p,p), mode="reflect")
    Hp, Wp = H + 2*p, W + 2*p
    targets = [x_gt]
    current = x_pad
    for tt in range(T):
        otf_t = gaussian_otf(delta, Hp, Wp)
        current = fft_conv2d_circular(current, otf_t)
        targets.append(current[:, :, p:p+H, p:p+W])

    # forward with precomputed targets (training mode)
    with torch.no_grad():
        result = model(y, sigma, None, targets)

    assert result["pred"].shape == y.shape
    assert len(result["stage_outputs"]) == T
    assert len(result["stage_targets"]) == T
    assert not torch.isnan(result["pred"]).any()

    # test stagewise loss
    base_loss = build_combined_loss([{"name": "l1"}])
    criterion = StagewiseLoss(T=T, base_loss=base_loss, learnable=False)
    loss, info = criterion(result["stage_outputs"], result["stage_targets"])
    assert not torch.isnan(loss)
    assert len(info["per_stage_loss"]) == T
    assert len(info["weights"]) == T

    # forward without targets (inference mode)
    with torch.no_grad():
        result2 = model(y, sigma)
    assert result2["stage_targets"] is None
    assert result2["pred"].shape == y.shape

    print(f"  ✓ {solver:5s} + {schedule:10s} + {denoiser:10s}  "
          f"loss={loss.item():.4f}  weights={[f'{w:.3f}' for w in info['weights']]}")


if __name__ == "__main__":
    solvers = ["hqs", "admm", "pg"]
    schedules = ["uniform", "trainable"]
    denoisers = ["dncnn", "unet_small", "resblock"]

    print("Running smoke tests (forward + stagewise loss)...")
    for sol in solvers:
        for sch in schedules:
            for den in denoisers:
                test_forward(sol, sch, den)

    print("\nAll tests passed! ✓")