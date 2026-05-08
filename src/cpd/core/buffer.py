"""Trajectory buffer — B+ (success) and B- (failure). Paper §3.1."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import torch

from cpd.core.trajectory import Trajectory


@dataclass
class TrajectoryBuffer:
    positives: list[Trajectory] = field(default_factory=list)
    negatives: list[Trajectory] = field(default_factory=list)

    def add(self, traj: Trajectory) -> None:
        if not traj.labeled:
            raise ValueError(
                "Cannot add unlabeled trajectory to buffer — call Labeler first."
            )
        bucket = self.positives if traj.success else self.negatives
        bucket.append(traj)

    def extend(self, trajs: Iterable[Trajectory]) -> None:
        for t in trajs:
            self.add(t)

    @property
    def n_positive(self) -> int:
        return len(self.positives)

    @property
    def n_negative(self) -> int:
        return len(self.negatives)

    def __len__(self) -> int:
        return self.n_positive + self.n_negative

    def __iter__(self) -> Iterator[Trajectory]:
        yield from self.positives
        yield from self.negatives

    def all_latents(self, *, positive: bool) -> torch.Tensor:
        """Stack (T_i, d) latents from B± into one (sum_T, d) tensor. Used by KDE (§3.4)."""
        bucket = self.positives if positive else self.negatives
        if not bucket:
            return torch.empty(0, 0)
        return torch.cat([t.latents for t in bucket], dim=0)
