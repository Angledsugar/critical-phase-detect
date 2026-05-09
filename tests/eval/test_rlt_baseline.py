"""RLTBaseline supervised handoff classifier."""
from __future__ import annotations

import torch

from cpd.core.trajectory import Trajectory
from cpd.eval.rlt_baseline import RLTBaseline


def _synthetic_trajectory(
    T: int, d: int, handoff_step: int, *, seed: int = 0
) -> Trajectory:
    """Trajectory whose latents have a clear regime change at `handoff_step`.

    Pre-handoff latents are drawn around -1; post-handoff around +1. So a
    classifier that sees the regime boundary should easily detect handoff.
    """
    g = torch.Generator().manual_seed(seed)
    latents = torch.zeros(T, d)
    latents[:handoff_step] = -1.0 + 0.05 * torch.randn(handoff_step, d, generator=g)
    latents[handoff_step:] = 1.0 + 0.05 * torch.randn(T - handoff_step, d, generator=g)
    return Trajectory(
        raw_states=list(range(T)),
        latents=latents,
        actions=tuple(range(T)),
        metadata={"handoff_step": handoff_step},
    )


def test_train_loss_decreases():
    torch.manual_seed(0)
    T, d = 30, 8
    trajs = [_synthetic_trajectory(T, d, handoff_step=15, seed=i) for i in range(6)]
    labels = {i: 15 for i in range(len(trajs))}
    model = RLTBaseline(window=8, hidden_dim=32)
    losses = model.fit(trajs, labels, epochs=15, lr=5e-3)
    assert len(losses) == 15
    # Loss must end strictly lower than the first epoch.
    assert losses[-1] < losses[0]


def test_predict_returns_int_in_range():
    torch.manual_seed(0)
    T, d = 20, 4
    trajs = [_synthetic_trajectory(T, d, handoff_step=10, seed=i) for i in range(3)]
    labels = {i: 10 for i in range(len(trajs))}
    model = RLTBaseline(window=4, hidden_dim=16)
    model.fit(trajs, labels, epochs=2, lr=1e-2)
    pred = model.predict(trajs[0])
    assert isinstance(pred, int)
    assert 0 <= pred < T


def test_predict_all_returns_dict():
    torch.manual_seed(0)
    T, d = 20, 4
    trajs = [_synthetic_trajectory(T, d, handoff_step=10, seed=i) for i in range(3)]
    labels = {i: 10 for i in range(3)}
    model = RLTBaseline(window=4, hidden_dim=16)
    model.fit(trajs, labels, epochs=2, lr=1e-2)
    out = model.predict_all(trajs)
    assert set(out.keys()) == {0, 1, 2}
    for v in out.values():
        assert isinstance(v, int)


def test_compute_implements_metric():
    torch.manual_seed(0)
    T, d = 20, 4
    trajs = [_synthetic_trajectory(T, d, handoff_step=10, seed=i) for i in range(3)]
    labels = {i: 10 for i in range(3)}
    model = RLTBaseline(window=4, hidden_dim=16)
    model.fit(trajs, labels, epochs=2, lr=1e-2)
    preds = model.predict_all(trajs)
    out = model.compute(preds, labels)
    assert "f1" in out
    assert "precision" in out
    assert "recall" in out
    assert "mean_step_error" in out


def test_predict_uninitialized_raises():
    import pytest

    model = RLTBaseline(window=4, hidden_dim=8)
    traj = _synthetic_trajectory(5, 4, handoff_step=2)
    with pytest.raises(RuntimeError):
        model.predict(traj)


def test_window_padding_for_short_traj():
    """Trajectory shorter than window must still produce per-step logits."""
    torch.manual_seed(0)
    model = RLTBaseline(window=8, hidden_dim=8)
    traj = _synthetic_trajectory(T=4, d=4, handoff_step=2)
    model.fit([traj], {0: 2}, epochs=1, lr=1e-3)
    pred = model.predict(traj)
    assert 0 <= pred < 4
