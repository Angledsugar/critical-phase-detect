"""Reward — finite, signed by buffer region, trajectory == sum of per-step."""
import math

import pytest
import torch

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.kde import compute_kde
from cpd.core.reward import Reward
from cpd.core.trajectory import Trajectory


def _labeled(latents: torch.Tensor, success: bool) -> Trajectory:
    T = latents.shape[0]
    return Trajectory(
        raw_states=list(range(T)), latents=latents, actions=(), success=success
    )


@pytest.fixture
def reward_with_buckets():
    torch.manual_seed(0)
    pos = torch.randn(40, 2) * 0.1 + torch.tensor([1.0, 1.0])  # cluster around +1
    neg = torch.randn(40, 2) * 0.1 + torch.tensor([-1.0, -1.0])  # cluster around -1
    buf = TrajectoryBuffer()
    buf.add(_labeled(pos, success=True))
    buf.add(_labeled(neg, success=False))
    kde = compute_kde(buf)
    return Reward(kde=kde)


def test_per_step_finite_batch(reward_with_buckets):
    z = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
    r = reward_with_buckets.per_step(z)
    assert r.shape == (3,)
    assert torch.all(torch.isfinite(r))


def test_per_step_single_point_returns_scalar(reward_with_buckets):
    r = reward_with_buckets.per_step(torch.tensor([1.0, 1.0]))
    assert r.shape == ()


def test_per_step_positive_in_demo_region(reward_with_buckets):
    r_pos_region = reward_with_buckets.per_step(torch.tensor([1.0, 1.0]))
    assert float(r_pos_region.item()) > 0.0


def test_per_step_negative_in_failure_region(reward_with_buckets):
    r_neg_region = reward_with_buckets.per_step(torch.tensor([-1.0, -1.0]))
    assert float(r_neg_region.item()) < 0.0


def test_trajectory_equals_sum_per_step(reward_with_buckets):
    torch.manual_seed(7)
    latents = torch.randn(6, 2)
    traj = Trajectory(raw_states=list(range(6)), latents=latents, actions=())
    expected = float(reward_with_buckets.per_step(latents).sum().item())
    got = reward_with_buckets.trajectory(traj)
    assert math.isclose(got, expected, rel_tol=1e-5, abs_tol=1e-5)
