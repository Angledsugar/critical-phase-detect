"""Reward r_t = r^(A) + R. Paper §3.5.

r^(A): per-step density-difference reward at φ(s_t), using KDE f̃_±.
R: trajectory-level success/failure weight applied at the terminal step.
Both terms derive from buffer statistics — no external hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from cpd.core.kde import KDEStats
from cpd.core.trajectory import Trajectory


@dataclass
class Reward:
    kde: KDEStats

    def per_step(self, latent: torch.Tensor) -> torch.Tensor:
        """r^(A) at latents of shape (B, d). Returns (B,)."""
        raise NotImplementedError("PR0 stub — implement in PR1.")

    def trajectory(self, traj: Trajectory) -> float:
        """R for a labeled trajectory (terminal-step weight)."""
        raise NotImplementedError("PR0 stub — implement in PR1.")
