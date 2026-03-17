"""Solver registry."""

from __future__ import annotations
import torch.nn as nn

from .base import BaseSolver
from .hqs import HQSSolver
from .admm import ADMMSolver
from .pg import PGSolver
from .ista import ISTASolver
from .fista import FISTASolver

SOLVER_REGISTRY: dict[str, type] = {
    "hqs": HQSSolver,
    "admm": ADMMSolver,
    "pg": PGSolver,
    "ista": ISTASolver,
    "fista": FISTASolver,
}


def build_solver(name: str, **kwargs) -> BaseSolver:
    if name not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown solver '{name}'. Choose from {list(SOLVER_REGISTRY)}")
    return SOLVER_REGISTRY[name](**kwargs)