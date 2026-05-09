"""Reward r_t = log f+(z) - log f-(z). Paper §3.5.

Per-step density-difference reward at φ(s_t), using KDE f̃_±. Trajectory-level
reward sums per-step rewards across the trajectory's latents. Both terms
derive from buffer statistics — no external hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from cpd.core.kde import KDEStats
from cpd.core.trajectory import Trajectory

_EPS = 1e-12


@dataclass
class Reward:
    kde: KDEStats

    def per_step(self, latent: torch.Tensor) -> torch.Tensor:
        """log f+(z) - log f-(z) at latents of shape (B, d) or (d,)."""
        squeeze = False
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
            squeeze = True
        f_pos = self.kde.density(latent, positive=True)
        f_neg = self.kde.density(latent, positive=False)
        f_pos = torch.clamp(f_pos, min=_EPS)
        f_neg = torch.clamp(f_neg, min=_EPS)
        out = torch.log(f_pos) - torch.log(f_neg)
        return out.squeeze(0) if squeeze else out

    def trajectory(self, traj: Trajectory) -> float:
        """Sum of per-step rewards over traj.latents."""
        rewards = self.per_step(traj.latents)
        return float(rewards.sum().item())
