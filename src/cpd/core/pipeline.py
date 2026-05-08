"""End-to-end detector pipeline. Paper §3.7.

Orchestration: encode → label → buffer → KDE → reward → conf.
A single DetectorPipeline drives the entire self-supervised CPD loop and
exposes per-step / trajectory rewards to a Policy refinement loop.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.kde import KDEStats
from cpd.core.labeler import Labeler
from cpd.core.reward import Reward
from cpd.core.trajectory import Trajectory


@dataclass
class DetectorPipeline:
    labeler: Labeler
    buffer: TrajectoryBuffer
    kde: KDEStats | None = None
    reward: Reward | None = None

    def ingest(self, trajs: Iterable[Trajectory]) -> None:
        """Label trajs (with conf gating), push to buffer, refresh stats."""
        raise NotImplementedError("PR0 stub — implement in PR1.")

    def refresh_stats(self) -> None:
        """Recompute KDE + reward from the current buffer."""
        raise NotImplementedError("PR0 stub — implement in PR1.")
