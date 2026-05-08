"""Trajectory — atomic data unit. Paper §3.1.

Carries raw env-specific states, φ-encoded latents, actions, and optional
labels (success / conf) produced by Labeler. Frozen by design: pipeline
stages return new Trajectory instances rather than mutating in place.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

RawState = Any
Action = Any


@dataclass(frozen=True)
class Trajectory:
    raw_states: Sequence[RawState]
    latents: torch.Tensor
    actions: Sequence[Action] = ()
    success: bool | None = None
    conf: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latents.ndim != 2:
            raise ValueError(
                f"latents must be (T, d), got shape {tuple(self.latents.shape)}"
            )
        T = self.latents.shape[0]
        if len(self.raw_states) != T:
            raise ValueError(
                f"raw_states length {len(self.raw_states)} != latents T={T}"
            )
        if len(self.actions) > 0 and len(self.actions) != T:
            raise ValueError(
                f"actions length {len(self.actions)} != latents T={T}"
            )

    def __len__(self) -> int:
        return self.latents.shape[0]

    @property
    def latent_dim(self) -> int:
        return self.latents.shape[1]

    @property
    def final_latent(self) -> torch.Tensor:
        """φ(s_T). Used by G2 labeler for goal/threshold derivation (§3.2)."""
        return self.latents[-1]

    @property
    def labeled(self) -> bool:
        return self.success is not None

    def with_label(self, *, success: bool, conf: float | None = None) -> Trajectory:
        return Trajectory(
            raw_states=self.raw_states,
            latents=self.latents,
            actions=self.actions,
            success=success,
            conf=conf,
            metadata=self.metadata,
        )
