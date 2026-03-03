"""Solver registry."""

from __future__ import annotations
import torch.nn as nn

from .base import BaseSolver
from .hqs import HQSSolver
from .admm import ADMMSolver
from .pg import PGSolver

SOLVER_REGISTRY: dict[str, type] = {
    "hqs": HQSSolver,
    "admm": ADMMSolver,
    "pg": PGSolver,
}


def build_solver(name: str, **kwargs) -> BaseSolver:
    if name not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown solver '{name}'. Choose from {list(SOLVER_REGISTRY)}")
    return SOLVER_REGISTRY[name](**kwargs)