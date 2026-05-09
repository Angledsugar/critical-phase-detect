"""TLDR triplet trainer. Paper §3.3.

Triplet sampling: for each anchor s_t along an expert demo τ_i, we draw
- positive  s_{t+k_pos}    (small temporal gap, same trajectory)
- negative  s_{t+K_neg}    (large temporal gap, same trajectory)

and minimize the margin-ranking loss
    L = mean( relu( ||f(a)-f(p)||² − ||f(a)-f(n)||² + margin ) ).

Indices are clamped to trajectory bounds, which biases sampling toward the
interior. K_neg ≫ k_pos ensures the negative is genuinely far in time.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from cpd.encoders.tldr import TLDREncoder, _to_tensor


def _trajectory_states_tensor(
    traj: Any, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Stack a trajectory's raw_states into a (T, state_dim) tensor."""
    raw = traj.raw_states if hasattr(traj, "raw_states") else traj
    if isinstance(raw, torch.Tensor):
        return raw.to(device=device, dtype=dtype)
    if isinstance(raw, np.ndarray):
        return torch.from_numpy(raw).to(device=device, dtype=dtype)
    rows = [_to_tensor(s, device=device, dtype=dtype) for s in raw]
    return torch.stack(rows, dim=0)


class TLDRTrainer:
    """Triplet-loss trainer for TLDREncoder."""

    def __init__(
        self,
        encoder: TLDREncoder,
        lr: float = 1e-3,
        k_pos: int = 2,
        K_neg: int = 20,
        margin: float = 1.0,
        batch_size: int = 256,
        weight_decay: float = 0.0,
        device: str | torch.device | None = None,
    ) -> None:
        if k_pos < 1:
            raise ValueError(f"k_pos must be >= 1, got {k_pos}")
        if K_neg <= k_pos:
            raise ValueError(f"K_neg ({K_neg}) must exceed k_pos ({k_pos})")
        self.encoder = encoder
        self.k_pos = int(k_pos)
        self.K_neg = int(K_neg)
        self.margin = float(margin)
        self.batch_size = int(batch_size)
        self.device = torch.device(device) if device is not None else next(
            encoder.parameters()
        ).device
        self.encoder.to(self.device)
        self.optimizer = torch.optim.Adam(
            encoder.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _sample_triplets(
        self, traj_states: list[torch.Tensor], rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Sample (anchor, pos, neg) batch from list of (T_i, state_dim) tensors.

        Anchors are picked uniformly across all valid timesteps. Positives are
        anchor index + k_pos (clamped). Negatives are anchor index + K_neg
        (clamped). Trajectories with T < 2 contribute nothing.
        """
        traj_lens = [t.shape[0] for t in traj_states]
        valid_idx = [i for i, T in enumerate(traj_lens) if T >= 2]
        if not valid_idx:
            return None

        anchors_list, pos_list, neg_list = [], [], []
        remaining = self.batch_size
        # weighted by trajectory length so longer demos contribute proportionally
        weights = np.array([traj_lens[i] for i in valid_idx], dtype=np.float64)
        weights = weights / weights.sum()
        while remaining > 0:
            i = valid_idx[int(rng.choice(len(valid_idx), p=weights))]
            states = traj_states[i]
            T = states.shape[0]
            t = int(rng.integers(0, T))
            t_pos = min(t + self.k_pos, T - 1)
            t_neg = min(t + self.K_neg, T - 1)
            if t_pos == t or t_neg == t_pos:
                # degenerate triplet — try again
                continue
            anchors_list.append(states[t])
            pos_list.append(states[t_pos])
            neg_list.append(states[t_neg])
            remaining -= 1
        a = torch.stack(anchors_list, dim=0)
        p = torch.stack(pos_list, dim=0)
        n = torch.stack(neg_list, dim=0)
        return a, p, n

    def step(
        self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor
    ) -> dict[str, float]:
        self.encoder.train()
        za = self.encoder(anchors)
        zp = self.encoder(positives)
        zn = self.encoder(negatives)
        d_ap = (za - zp).pow(2).sum(dim=-1)
        d_an = (za - zn).pow(2).sum(dim=-1)
        diff = d_ap - d_an + self.margin
        loss = nn.functional.relu(diff).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            violation_rate = (diff > 0).float().mean().item()
        return {
            "loss": loss.item(),
            "violation_rate": violation_rate,
            "d_ap": d_ap.mean().item(),
            "d_an": d_an.mean().item(),
        }

    def train_epoch(
        self,
        trajectories: Sequence[Any],
        *,
        num_batches: int = 1,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Run num_batches triplet updates over the given trajectories.

        trajectories: sequence of objects with `.raw_states` (or raw arrays).
        """
        rng = np.random.default_rng(seed)
        param = next(self.encoder.parameters())
        traj_states = [
            _trajectory_states_tensor(t, device=param.device, dtype=param.dtype)
            for t in trajectories
        ]
        agg: dict[str, float] = {"loss": 0.0, "violation_rate": 0.0,
                                 "d_ap": 0.0, "d_an": 0.0}
        steps = 0
        for _ in range(num_batches):
            sample = self._sample_triplets(traj_states, rng)
            if sample is None:
                break
            metrics = self.step(*sample)
            for k, v in metrics.items():
                agg[k] += v
            steps += 1
        if steps == 0:
            return {k: float("nan") for k in agg}
        return {k: v / steps for k, v in agg.items()}
