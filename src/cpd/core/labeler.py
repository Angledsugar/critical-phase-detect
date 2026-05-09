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
    """Maps a trajectory to a labeled trajectory."""

    def label(self, traj: Trajectory) -> Trajectory: ...


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
        if len(demos) == 0:
            raise ValueError("from_demos requires at least one demo trajectory")
        if not (0.0 < quantile <= 1.0):
            raise ValueError(f"quantile must be in (0, 1], got {quantile}")
        finals = torch.stack([t.final_latent for t in demos], dim=0)
        g = finals.mean(dim=0)
        dists = torch.linalg.norm(finals - g, dim=-1)
        epsilon = float(torch.quantile(dists, quantile).item())
        return cls(g=g, epsilon=epsilon)

    def label(self, traj: Trajectory) -> Trajectory:
        dist = float(torch.linalg.norm(traj.final_latent - self.g).item())
        success = dist <= self.epsilon
        return traj.with_label(success=success)
