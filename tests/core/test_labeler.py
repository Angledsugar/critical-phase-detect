"""G2Labeler — from_demos derivation, label of in/out points."""
import pytest
import torch

from cpd.core.labeler import G2Labeler
from cpd.core.trajectory import Trajectory


def _demo(final: torch.Tensor, T: int = 5) -> Trajectory:
    d = final.shape[0]
    latents = torch.zeros(T, d)
    latents[-1] = final
    return Trajectory(raw_states=list(range(T)), latents=latents, actions=())


def test_from_demos_derives_g_and_eps():
    finals = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32
    )
    demos = [_demo(f) for f in finals]
    g2 = G2Labeler.from_demos(demos, quantile=0.95)
    assert torch.allclose(g2.g, finals.mean(dim=0), atol=1e-6)
    # epsilon is a quantile of distances ⇒ within [min, max] dists
    dists = torch.linalg.norm(finals - g2.g, dim=-1)
    assert float(dists.min().item()) - 1e-6 <= g2.epsilon <= float(dists.max().item()) + 1e-6


def test_from_demos_empty_raises():
    with pytest.raises(ValueError):
        G2Labeler.from_demos([], quantile=0.95)


def test_from_demos_invalid_quantile():
    finals = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    demos = [_demo(f) for f in finals]
    with pytest.raises(ValueError):
        G2Labeler.from_demos(demos, quantile=0.0)
    with pytest.raises(ValueError):
        G2Labeler.from_demos(demos, quantile=1.5)


def test_label_demo_is_positive():
    finals = torch.tensor([[1.0, 0.0], [0.9, 0.1], [1.1, -0.1]], dtype=torch.float32)
    demos = [_demo(f) for f in finals]
    g2 = G2Labeler.from_demos(demos, quantile=1.0)
    # Any demo's final equals an existing demo final ⇒ within max dist ⇒ positive.
    test_traj = _demo(finals[0])
    out = g2.label(test_traj)
    assert out.success is True
    assert out.labeled


def test_label_far_point_is_negative():
    finals = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0], [0.0, 0.1]], dtype=torch.float32
    )
    demos = [_demo(f) for f in finals]
    g2 = G2Labeler.from_demos(demos, quantile=0.95)
    far = _demo(torch.tensor([10.0, 10.0]))
    out = g2.label(far)
    assert out.success is False


def test_label_returns_new_trajectory():
    finals = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    demos = [_demo(f) for f in finals]
    g2 = G2Labeler.from_demos(demos, quantile=0.95)
    traj = _demo(torch.tensor([0.0, 0.5]))
    labeled = g2.label(traj)
    assert traj.success is None  # original untouched
    assert labeled.success is not None
