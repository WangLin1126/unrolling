"""Experiment directory management.

Builds structured output paths:
  results/{dataset}/{param_summary}/{timestamp}/train/
  results/{dataset}/{param_summary}/{timestamp}/test/
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path


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
        f"-schedule_{'learn' if mc.get('learnable_schedule') else mc['schedule']}"
        f"-lossw_{'learn' if mc.get('learnable_loss_weights') else 'uniform'}"
        f"-lmode_{tc.get('loss_mode')}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return Path(base) / dataset_name / params / timestamp