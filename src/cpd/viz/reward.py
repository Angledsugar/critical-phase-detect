"""Reward-curve visualisation. Paper §3.5 / Fig 4."""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from cpd.core.trajectory import Trajectory


def plot_reward_curve(traj: Trajectory, reward_fn: Any) -> Figure:
    """Per-step reward over a trajectory.

    `reward_fn` is anything with a `.per_step(latents) -> torch.Tensor` API
    (e.g. `cpd.core.reward.Reward`). If `traj.metadata` carries a `handoff_step`
    we mark it with a vertical dashed line.
    """
    latents = traj.latents
    rewards = reward_fn.per_step(latents)
    rewards_np = (
        rewards.detach().cpu().numpy() if isinstance(rewards, torch.Tensor) else rewards
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    steps = list(range(len(rewards_np)))
    ax.plot(steps, rewards_np, color="C0", linewidth=1.4, label="r^(A)")
    ax.set_xlabel("step")
    ax.set_ylabel("per-step reward")

    handoff = traj.metadata.get("handoff_step") if traj.metadata else None
    if handoff is not None:
        ax.axvline(
            float(handoff),
            color="C3",
            linestyle="--",
            linewidth=1.2,
            label=f"handoff @ {int(handoff)}",
        )
    ax.set_title("Per-step reward")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


__all__ = ["plot_reward_curve"]
