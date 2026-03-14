"""Abstract base class for unrolling solvers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Union

import torch
import torch.nn as nn


DenoiserLike = Union[nn.Module, Callable[..., torch.Tensor]]


class BaseSolver(nn.Module, ABC):
    """One stage of the unrolled reverse chain: given x_t, produce x_{t-1}."""

    @abstractmethod
    def step(
        self,
        x_t: torch.Tensor,
        denoiser: DenoiserLike,
        otf: torch.Tensor,
        beta: torch.Tensor,
        inner_iters: int = 1,
        noise_sigma = None,
    ) -> torch.Tensor:
        ...
