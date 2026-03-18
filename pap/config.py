"""PAP configuration parsing and validation."""

from __future__ import annotations
from pathlib import Path
from typing import Any

from models.denoisers import DENOISER_REGISTRY


def parse_denoiser_chain(chain_cfg: list[dict]) -> list[dict]:
    """Parse and sort the denoiser chain config by position.

    Args:
        chain_cfg: list of denoiser entries from YAML, each with keys:
            type, position, pretrain, trainable, params

    Returns:
        Sorted list of denoiser configs (by position, ascending).

    Raises:
        ValueError: on duplicate positions or unknown denoiser types.
    """
    if not chain_cfg:
        raise ValueError("denoiser_chain cannot be empty")

    # Validate and collect
    entries = []
    positions_seen = set()
    for i, entry in enumerate(chain_cfg):
        dtype = entry.get("type")
        if dtype not in DENOISER_REGISTRY:
            raise ValueError(
                f"Entry {i}: unknown denoiser type '{dtype}'. "
                f"Available: {list(DENOISER_REGISTRY)}"
            )

        pos = entry.get("position")
        if pos is None:
            raise ValueError(f"Entry {i} (type='{dtype}'): 'position' is required")
        pos = int(pos)

        if pos in positions_seen:
            raise ValueError(
                f"Entry {i} (type='{dtype}'): duplicate position {pos}. "
                f"Each denoiser must have a unique position."
            )
        positions_seen.add(pos)

        pretrain = entry.get("pretrain", None)
        if pretrain is not None and isinstance(pretrain, str):
            pretrain = pretrain.strip()
            if pretrain.lower() in ("null", "none", ""):
                pretrain = None

        trainable = entry.get("trainable", True)
        params = dict(entry.get("params", {}))

        entries.append({
            "type": dtype,
            "position": pos,
            "pretrain": pretrain,
            "trainable": bool(trainable),
            "params": params,
        })

    # Sort by position (relative ordering)
    entries.sort(key=lambda e: e["position"])
    return entries


def validate_chain_config(chain: list[dict], in_channels: int = 3) -> None:
    """Validate that the denoiser chain is self-consistent.

    Checks:
        - All denoiser types exist in registry
        - Pretrained checkpoint paths exist (if specified)
        - No duplicate positions
        - in_channels consistency

    Raises:
        ValueError: on validation failure.
        FileNotFoundError: if a pretrain path doesn't exist.
    """
    positions = [e["position"] for e in chain]
    if len(positions) != len(set(positions)):
        raise ValueError(f"Duplicate positions found: {positions}")

    for i, entry in enumerate(chain):
        dtype = entry["type"]
        if dtype not in DENOISER_REGISTRY:
            raise ValueError(f"Stage {i}: unknown denoiser '{dtype}'")

        pretrain = entry.get("pretrain")
        if pretrain is not None:
            p = Path(pretrain)
            if not p.exists():
                raise FileNotFoundError(
                    f"Stage {i} (type='{dtype}'): pretrain path not found: {pretrain}"
                )

    # Check that all stages have valid params (try to instantiate)
    for i, entry in enumerate(chain):
        cls = DENOISER_REGISTRY[entry["type"]]
        try:
            params = dict(entry["params"])
            params["in_channels"] = in_channels
            model = cls(**params)
            del model
        except Exception as e:
            raise ValueError(
                f"Stage {i} (type='{entry['type']}'): "
                f"invalid params {entry['params']}: {e}"
            ) from e
