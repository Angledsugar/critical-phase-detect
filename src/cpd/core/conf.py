"""OOD reliability — conf metric + threshold. Paper §3.6.

For a trajectory whose final latent is far from any seen demo / buffer point,
the labeler's binary verdict is unreliable. conf ∈ [0, 1] scores reliability;
below a (buffer-derived) threshold the trajectory is excluded from B±.
"""
from __future__ import annotations

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.trajectory import Trajectory


def compute_conf(traj: Trajectory, buf: TrajectoryBuffer) -> float:
    """Reliability of labeler output for traj given current buffer state."""
    raise NotImplementedError("PR0 stub — implement in PR1.")


def conf_threshold(buf: TrajectoryBuffer) -> float:
    """Auto-derived threshold from buffer occupancy."""
    raise NotImplementedError("PR0 stub — implement in PR1.")
