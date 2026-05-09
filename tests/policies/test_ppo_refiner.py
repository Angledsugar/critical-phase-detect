"""AdapterHead + Pi0RefinedPolicy unit tests (no openpi dependency)."""

from __future__ import annotations

import torch

from cpd.core.trajectory import Trajectory
from cpd.policies.ppo_refiner import AdapterHead, Pi0RefinedPolicy, PPOConfig


class _StubBase:
    """Constant base policy: predict() returns a fixed action."""

    def __init__(self, action_dim: int) -> None:
        self.action = torch.zeros(action_dim) + 0.3

    def predict(self, obs):  # noqa: ARG002
        return self.action.clone()

    def refine_step(self, batch):  # noqa: ARG002
        return {}


class _StubReward:
    """reward.per_step(latents (T, d)) -> (T,) dummy scalars."""

    def per_step(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.mean(dim=-1)


def _make_traj(T: int = 6, d: int = 16, action_dim: int = 4) -> Trajectory:
    return Trajectory(
        raw_states=list(range(T)),
        latents=torch.randn(T, d),
        actions=tuple(torch.randn(action_dim) for _ in range(T)),
    )


def test_adapter_forward_shape():
    ad = AdapterHead(in_dim=16, action_dim=4, rank=8)
    out = ad(torch.randn(16))
    assert out.shape == (4,)


def test_adapter_zero_init_matches_base():
    ad = AdapterHead(in_dim=16, action_dim=4, rank=8)
    feat = torch.randn(16)
    base_action = torch.full((4,), 0.5)
    refined = base_action + ad(feat)
    torch.testing.assert_close(refined, base_action)


def test_refined_policy_predict_uses_base_initially():
    ad = AdapterHead(in_dim=16, action_dim=4, rank=8)
    base = _StubBase(action_dim=4)
    pol = Pi0RefinedPolicy(base=base, adapter=ad, reward_fn=_StubReward())
    obs = torch.randn(16)
    out = pol.predict(obs)
    torch.testing.assert_close(out, base.action)


def test_refine_step_no_nan_and_changes_params():
    torch.manual_seed(0)
    ad = AdapterHead(in_dim=16, action_dim=4, rank=8)
    base = _StubBase(action_dim=4)
    pol = Pi0RefinedPolicy(
        base=base,
        adapter=ad,
        reward_fn=_StubReward(),
        config=PPOConfig(lr=1e-2, epochs=2),
    )
    traj = _make_traj(T=8, d=16, action_dim=4)

    before_down = ad.down.weight.detach().clone()
    before_up = ad.up.weight.detach().clone()

    metrics = pol.refine_step(traj)

    for v in metrics.values():
        assert not (isinstance(v, float) and (v != v)), f"NaN metric: {metrics}"

    delta = (ad.down.weight - before_down).abs().sum() + (ad.up.weight - before_up).abs().sum()
    assert delta.item() > 0.0, "adapter params did not change after refine_step"


def test_refine_step_empty_trajectory():
    ad = AdapterHead(in_dim=16, action_dim=4, rank=8)
    base = _StubBase(action_dim=4)
    pol = Pi0RefinedPolicy(base=base, adapter=ad, reward_fn=_StubReward())
    empty = Trajectory(raw_states=[], latents=torch.zeros(0, 16), actions=())
    metrics = pol.refine_step(empty)
    assert metrics.get("skipped") == 1.0
