"""RLT (RL Token) supervised baseline. Paper §3.1 / §6.2.

Predicts per-step handoff probability from a sliding window of latents.
Trained against per-trajectory single handoff-step labels — at training time
each trajectory contributes a binary target sequence (1 at the handoff step,
0 elsewhere) with optional smoothing.

Architecture: a small 1-D CNN (Conv1d → ReLU → Conv1d → Sigmoid) over a
window of latents. We take an explicit window because RLT's published recipe
classifies critical phase from a short context, not the entire trajectory.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from cpd.core.trajectory import Trajectory
from cpd.eval.metrics import F1Metric


class RLTBaseline(nn.Module):
    """Supervised handoff-step classifier over a sliding window of latents.

    Args:
        window: number of latent steps fed into each per-step prediction.
        hidden_dim: width of the 1-D CNN's hidden channel.
        latent_dim: dimensionality of φ. Inferred on first .fit() call if -1.
    """

    def __init__(
        self,
        window: int = 8,
        hidden_dim: int = 64,
        latent_dim: int = -1,
        smoothing: int = 1,
    ) -> None:
        super().__init__()
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.window = window
        self.hidden_dim = hidden_dim
        # WHY smoothing: a single positive label is too sparse — broaden the
        # target by ±smoothing steps so gradient signal isn't trivially zero.
        self.smoothing = smoothing
        self._latent_dim = latent_dim
        self.name = "rlt_baseline"
        self.net: nn.Module | None = None
        if latent_dim > 0:
            self._build(latent_dim)

    def _build(self, latent_dim: int) -> None:
        self._latent_dim = latent_dim
        # Conv1d expects (B, C, L); we use C = latent_dim, L = window.
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _windows(self, latents: torch.Tensor) -> torch.Tensor:
        """Build per-step windows. (T, d) -> (T, d, window) (left-padded)."""
        T, d = latents.shape
        # WHY left-pad with the first latent: a per-step prediction at step t
        # must use latents from [t - window + 1, t]; for t < window - 1 we
        # repeat the first frame to fill the gap.
        pad_count = self.window - 1
        if pad_count > 0:
            pad = latents[:1].expand(pad_count, d)
            padded = torch.cat([pad, latents], dim=0)
        else:
            padded = latents
        # Sliding view: padded[i : i + window] for i in [0, T)
        windows = torch.stack(
            [padded[i : i + self.window] for i in range(T)], dim=0
        )
        # (T, window, d) -> (T, d, window) for Conv1d.
        return windows.transpose(1, 2)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Return per-step logits of shape (T,) for one trajectory's latents."""
        if self.net is None:
            raise RuntimeError(
                "RLTBaseline.net is uninitialized — call .fit() or pass latent_dim "
                "to __init__."
            )
        if latents.ndim != 2:
            raise ValueError(
                f"latents must be (T, d); got shape {tuple(latents.shape)}"
            )
        windows = self._windows(latents)  # (T, d, window)
        logits = self.net(windows).squeeze(-1)  # (T,)
        return logits

    def fit(
        self,
        trajectories: Sequence[Trajectory],
        labels: dict[Any, int],
        *,
        epochs: int = 20,
        lr: float = 1e-3,
        traj_ids: Sequence[Any] | None = None,
    ) -> list[float]:
        """Supervised training. Returns list of mean train losses per epoch."""
        if not trajectories:
            raise ValueError("Empty trajectories.")
        # Build trajectory IDs: defaults to enumeration index.
        ids = list(traj_ids) if traj_ids is not None else list(range(len(trajectories)))
        if len(ids) != len(trajectories):
            raise ValueError("len(traj_ids) must match len(trajectories).")

        # Lazy build: infer latent_dim from data.
        d = trajectories[0].latent_dim
        if self.net is None:
            self._build(d)
        elif d != self._latent_dim:
            raise ValueError(
                f"Latent dim mismatch: model expects {self._latent_dim}, got {d}."
            )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()

        # Pre-compute targets so we don't re-build every epoch.
        targets: list[torch.Tensor] = []
        active: list[int] = []  # indices of trajectories that have a label
        for i, (tid, traj) in enumerate(zip(ids, trajectories, strict=False)):
            if tid not in labels:
                continue
            T = len(traj)
            y = torch.zeros(T)
            step = int(labels[tid])
            lo = max(0, step - self.smoothing)
            hi = min(T, step + self.smoothing + 1)
            y[lo:hi] = 1.0
            targets.append(y)
            active.append(i)
        if not active:
            raise ValueError("No trajectories with matching labels.")

        losses: list[float] = []
        self.train()
        for _ in range(epochs):
            total = 0.0
            for idx, target in zip(active, targets, strict=False):
                latents = trajectories[idx].latents
                logits = self(latents)
                loss = bce(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.item())
            losses.append(total / len(active))
        return losses

    @torch.no_grad()
    def predict(self, trajectory: Trajectory) -> int:
        """Predicted handoff step. argmax of per-step probabilities."""
        if self.net is None:
            raise RuntimeError(
                "RLTBaseline is uninitialized — call .fit() before .predict()."
            )
        self.eval()
        logits = self(trajectory.latents)
        # WHY argmax (not first-crossing): logits may never cross 0 on tiny
        # / uncalibrated runs; argmax always returns a valid index.
        step = int(torch.argmax(logits).item())
        return step

    def predict_all(
        self,
        trajectories: Sequence[Trajectory],
        traj_ids: Sequence[Any] | None = None,
    ) -> dict[Any, int]:
        ids = list(traj_ids) if traj_ids is not None else list(range(len(trajectories)))
        return {tid: self.predict(t) for tid, t in zip(ids, trajectories, strict=False)}

    def compute(
        self,
        predictions: dict[Any, int],
        ground_truth: dict[Any, int],
    ) -> dict[str, float]:
        """Implements Metric protocol via F1Metric."""
        return F1Metric().compute(predictions, ground_truth)
