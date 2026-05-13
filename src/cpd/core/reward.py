"""Reward r_t = log f+(z) - log f-(z). Paper §3.5.

Per-step density-difference reward at φ(s_t), using KDE f̃_±. Trajectory-level
reward sums per-step rewards across the trajectory's latents. Both terms
derive from buffer statistics — no external hyperparameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cpd.core.kde import KDEStats
from cpd.core.trajectory import Trajectory

_EPS = 1e-12


@dataclass
class Reward:
    kde: KDEStats

    def per_step(self, latent: torch.Tensor) -> torch.Tensor:
        """log f+(z) - log f-(z) at latents of shape (B, d) or (d,).

        Computed directly in log-space so high-d KDE values don't over/underflow
        the exp/log roundtrip. -inf log-densities (empty buckets) are floored to
        log(_EPS) to keep r_t finite.
        """
        squeeze = False
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
            squeeze = True
        log_pos = self.kde.log_density(latent, positive=True)
        log_neg = self.kde.log_density(latent, positive=False)
        floor = math.log(_EPS)
        log_pos = torch.clamp(log_pos, min=floor)
        log_neg = torch.clamp(log_neg, min=floor)
        out = log_pos - log_neg
        return out.squeeze(0) if squeeze else out

    def trajectory(self, traj: Trajectory) -> float:
        """Sum of per-step rewards over traj.latents."""
        rewards = self.per_step(traj.latents)
        return float(rewards.sum().item())
