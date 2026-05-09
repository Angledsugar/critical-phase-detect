"""Kernel-weighted buffer statistics — f̃_+, f̃_-. Paper §3.4.

Gaussian KDE over latents in B+, B-. Bandwidth h derived from buffer
statistics (Silverman's rule). No external hyperparameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cpd.core.buffer import TrajectoryBuffer


@dataclass
class KDEStats:
    h: float
    latents_pos: torch.Tensor  # (sum_T_pos, d)
    latents_neg: torch.Tensor  # (sum_T_neg, d)

    def density(self, x: torch.Tensor, *, positive: bool) -> torch.Tensor:
        """Evaluate f̃_± at points x of shape (B, d) or (d,). Returns (B,) or scalar."""
        latents = self.latents_pos if positive else self.latents_neg
        squeeze = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze = True
        if latents.numel() == 0 or latents.shape[0] == 0:
            out = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            return out.squeeze(0) if squeeze else out

        d = x.shape[-1]
        n = latents.shape[0]
        h = float(self.h)
        # Squared distances (B, N)
        diff = x.unsqueeze(1) - latents.unsqueeze(0)
        sq = (diff * diff).sum(dim=-1)
        # log Gaussian kernel:  -0.5 * sq / h^2  - 0.5 d log(2π h^2)
        log_norm = -0.5 * d * math.log(2.0 * math.pi * h * h)
        log_kern = -0.5 * sq / (h * h) + log_norm
        # log(1/N) + logsumexp over neighbors
        log_density = torch.logsumexp(log_kern, dim=-1) - math.log(n)
        out = torch.exp(log_density)
        return out.squeeze(0) if squeeze else out


def silverman_bandwidth(latents: torch.Tensor) -> float:
    """Silverman's rule: h = (4/(d+2))^(1/(d+4)) · n^(-1/(d+4)) · σ̂. (n, d) → h."""
    if latents.ndim != 2:
        raise ValueError(f"latents must be (N, d), got shape {tuple(latents.shape)}")
    n, d = latents.shape
    if n < 2:
        raise ValueError(f"silverman_bandwidth requires N >= 2, got N={n}")
    std_per_dim = latents.std(dim=0, unbiased=True)
    sigma = float(std_per_dim.mean().item())
    if sigma <= 0.0 or not math.isfinite(sigma):
        sigma = 1e-6
    factor = (4.0 / (d + 2.0)) ** (1.0 / (d + 4.0))
    h = factor * (n ** (-1.0 / (d + 4.0))) * sigma
    return float(h)


def compute_kde(buf: TrajectoryBuffer) -> KDEStats:
    """Build KDEStats from buffer. Bandwidth auto-derived from union of B±."""
    pos = buf.all_latents(positive=True)
    neg = buf.all_latents(positive=False)
    if pos.numel() == 0 and neg.numel() == 0:
        raise ValueError("compute_kde requires a non-empty buffer")
    # Use union for bandwidth so h is consistent across f+ and f-.
    if pos.numel() > 0 and neg.numel() > 0:
        union = torch.cat([pos, neg], dim=0)
    elif pos.numel() > 0:
        union = pos
    else:
        union = neg
    h = silverman_bandwidth(union)
    return KDEStats(h=h, latents_pos=pos, latents_neg=neg)
