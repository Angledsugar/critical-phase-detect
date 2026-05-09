"""OOD reliability — conf metric + threshold. Paper §3.6.

For a trajectory whose final latent is far from any seen demo / buffer point,
the labeler's binary verdict is unreliable. conf ∈ [0, 1] scores reliability;
below a (buffer-derived) threshold the trajectory is excluded from B±.
"""
from __future__ import annotations

import math

import torch

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.trajectory import Trajectory


def _nearest_distance(point: torch.Tensor, pool: torch.Tensor) -> float:
    """Minimum L2 distance from `point` (d,) to any row of `pool` (N, d)."""
    if pool.numel() == 0 or pool.shape[0] == 0:
        return float("inf")
    diffs = pool - point.unsqueeze(0)
    dists = torch.linalg.norm(diffs, dim=-1)
    return float(dists.min().item())


def compute_conf(traj: Trajectory, buf: TrajectoryBuffer) -> float:
    """Reliability of labeler output: sigmoid(signed margin to opposite class).

    margin = d_opposite - d_same  (positive ⇒ trajectory's final latent is closer
    to the same-class neighborhood than to the opposite class). When the
    trajectory has no label, both buckets are queried symmetrically and conf
    falls back to 0.5 if either bucket is empty.
    """
    z = traj.final_latent
    d_pos = _nearest_distance(z, buf.all_latents(positive=True))
    d_neg = _nearest_distance(z, buf.all_latents(positive=False))

    # No information: cannot judge reliability.
    if math.isinf(d_pos) and math.isinf(d_neg):
        return 0.5
    if math.isinf(d_pos) or math.isinf(d_neg):
        # Only one bucket populated — fall back to neutral.
        return 0.5

    if traj.success is True:
        margin = d_neg - d_pos
    elif traj.success is False:
        margin = d_pos - d_neg
    else:
        margin = abs(d_pos - d_neg)

    return float(torch.sigmoid(torch.tensor(margin, dtype=torch.float32)).item())


def conf_threshold(buf: TrajectoryBuffer) -> float:
    """Adaptive threshold: median of computed confs across buffered trajectories.

    Returns 0.5 when fewer than 2 trajectories are available.
    """
    trajs = list(buf)
    if len(trajs) < 2:
        return 0.5
    confs = [compute_conf(t, buf) for t in trajs]
    return float(torch.tensor(confs, dtype=torch.float32).median().item())
