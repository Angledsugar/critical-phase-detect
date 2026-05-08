"""Labeler protocol + G2 self-supervised labeler. Paper §3.2.

G2 derivation (no external hyperparameters):
  g       = mean of demo final latents φ(s_T^(i))
  ε       = quantile (default 95%) of demo final-latent distances ‖φ(s_T^(i)) − g‖
A new trajectory τ is labeled success iff ‖φ(s_T(τ)) − g‖ < ε.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from cpd.core.trajectory import Trajectory


class Labeler(Protocol):
    """Maps a trajectory to (success, conf)."""

    def label(self, traj: Trajectory) -> tuple[bool, float]: ...


@dataclass
class G2Labeler:
    """Self-supervised labeler from expert demos. Paper §3.2."""

    g: torch.Tensor
    epsilon: float

    @classmethod
    def from_demos(
        cls,
        demos: Sequence[Trajectory],
        *,
        quantile: float = 0.95,
    ) -> G2Labeler:
        raise NotImplementedError("PR0 stub — implement in PR1 (T1 core impl).")

    def label(self, traj: Trajectory) -> tuple[bool, float]:
        raise NotImplementedError("PR0 stub — implement in PR1 (T1 core impl).")
