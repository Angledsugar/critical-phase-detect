"""End-to-end detector pipeline. Paper §3.7.

Orchestration: encode → label → buffer → KDE → reward → conf.
A single DetectorPipeline drives the entire self-supervised CPD loop and
exposes per-step / trajectory rewards to a Policy refinement loop.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.kde import KDEStats, compute_kde
from cpd.core.labeler import Labeler
from cpd.core.reward import Reward
from cpd.core.trajectory import Trajectory


@dataclass
class DetectorPipeline:
    labeler: Labeler
    buffer: TrajectoryBuffer
    kde: KDEStats | None = None
    reward: Reward | None = None

    def ingest(self, trajs: Trajectory | Iterable[Trajectory]) -> None:
        """Label trajs, push to buffer, refresh stats once both buckets exist."""
        if isinstance(trajs, Trajectory):
            trajs = [trajs]
        any_added = False
        for traj in trajs:
            labeled = traj if traj.labeled else self.labeler.label(traj)
            self.buffer.add(labeled)
            any_added = True
        if any_added and self.buffer.n_positive > 0 and self.buffer.n_negative > 0:
            self.refresh_stats()

    def refresh_stats(self) -> None:
        """Recompute KDE + reward from the current buffer. Idempotent."""
        if self.buffer.n_positive == 0 and self.buffer.n_negative == 0:
            return
        self.kde = compute_kde(self.buffer)
        self.reward = Reward(kde=self.kde)
