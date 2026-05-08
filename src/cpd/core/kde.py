"""Kernel-weighted buffer statistics — f̃_+, f̃_-. Paper §3.4.

Gaussian KDE over latents in B+, B-. Bandwidth h derived from buffer
statistics (Silverman's rule). No external hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from cpd.core.buffer import TrajectoryBuffer


@dataclass
class KDEStats:
    h: float
    latents_pos: torch.Tensor  # (sum_T_pos, d)
    latents_neg: torch.Tensor  # (sum_T_neg, d)

    def density(self, x: torch.Tensor, *, positive: bool) -> torch.Tensor:
        """Evaluate f̃_± at points x of shape (B, d). Returns (B,)."""
        raise NotImplementedError("PR0 stub — implement in PR1.")


def silverman_bandwidth(latents: torch.Tensor) -> float:
    """Silverman's rule: h = (4/(d+2))^(1/(d+4)) · n^(-1/(d+4)) · σ̂. (n, d) → h."""
    raise NotImplementedError("PR0 stub — implement in PR1.")


def compute_kde(buf: TrajectoryBuffer) -> KDEStats:
    """Build KDEStats from buffer. Bandwidth auto-derived."""
    raise NotImplementedError("PR0 stub — implement in PR1.")
