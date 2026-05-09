"""Plot-function smoke tests — each must return a Figure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from cpd.core.trajectory import Trajectory
from cpd.viz.handoff import plot_handoff_distribution
from cpd.viz.kde import plot_kde_2d
from cpd.viz.reward import plot_reward_curve


class _FakeKDEStats:
    def __init__(
        self, h: float, latents_pos: torch.Tensor, latents_neg: torch.Tensor
    ) -> None:
        self.h = h
        self.latents_pos = latents_pos
        self.latents_neg = latents_neg


class _FakeReward:
    """Reward stub: per_step returns the step index along latent dim 0."""

    def per_step(self, latents: torch.Tensor) -> torch.Tensor:
        return latents[:, 0]


def test_plot_kde_2d_returns_figure():
    torch.manual_seed(0)
    pos = torch.randn(20, 4) + torch.tensor([1.0, 1.0, 0.0, 0.0])
    neg = torch.randn(20, 4) + torch.tensor([-1.0, -1.0, 0.0, 0.0])
    stats = _FakeKDEStats(h=0.3, latents_pos=pos, latents_neg=neg)
    fig = plot_kde_2d(stats, axis_dims=(0, 1))
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_kde_2d_other_dims():
    torch.manual_seed(0)
    pos = torch.randn(10, 5)
    neg = torch.randn(10, 5)
    stats = _FakeKDEStats(h=0.4, latents_pos=pos, latents_neg=neg)
    fig = plot_kde_2d(stats, axis_dims=(2, 3))
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_kde_2d_empty():
    stats = _FakeKDEStats(
        h=0.1, latents_pos=torch.empty(0, 2), latents_neg=torch.empty(0, 2)
    )
    fig = plot_kde_2d(stats, axis_dims=(0, 1))
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_reward_curve_returns_figure():
    T, d = 20, 4
    traj = Trajectory(
        raw_states=list(range(T)),
        latents=torch.randn(T, d),
        actions=tuple(range(T)),
        metadata={"handoff_step": 10},
    )
    fig = plot_reward_curve(traj, _FakeReward())
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_reward_curve_no_handoff():
    T, d = 15, 3
    traj = Trajectory(
        raw_states=list(range(T)),
        latents=torch.randn(T, d),
        actions=tuple(range(T)),
    )
    fig = plot_reward_curve(traj, _FakeReward())
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_handoff_distribution_returns_figure():
    preds = {i: 5 + i for i in range(10)}
    gt = {i: 6 + i for i in range(10)}
    fig = plot_handoff_distribution(preds, gt)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_handoff_distribution_with_missing():
    preds = {0: 5, 1: None, 2: 10}
    gt = {0: 5, 1: 8, 2: 10, 3: 12}
    fig = plot_handoff_distribution(preds, gt)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_handoff_distribution_empty():
    fig = plot_handoff_distribution({}, {})
    assert isinstance(fig, Figure)
    plt.close(fig)
