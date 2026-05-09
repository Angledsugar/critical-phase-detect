"""conf — bounded in [0, 1], higher for clearer cases, neutral when underdetermined."""
import torch

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.conf import compute_conf, conf_threshold
from cpd.core.trajectory import Trajectory


def _labeled(latents: torch.Tensor, success: bool) -> Trajectory:
    T = latents.shape[0]
    return Trajectory(
        raw_states=list(range(T)), latents=latents, actions=(), success=success
    )


def test_conf_in_unit_interval():
    torch.manual_seed(0)
    buf = TrajectoryBuffer()
    buf.add(_labeled(torch.randn(5, 2), success=True))
    buf.add(_labeled(torch.randn(5, 2), success=False))
    traj = _labeled(torch.randn(3, 2), success=True)
    c = compute_conf(traj, buf)
    assert 0.0 <= c <= 1.0


def test_conf_higher_for_clearer_positive():
    pos = torch.zeros(1, 2) + torch.tensor([[1.0, 1.0]])
    neg = torch.zeros(1, 2) + torch.tensor([[-1.0, -1.0]])
    buf = TrajectoryBuffer()
    buf.add(_labeled(pos, success=True))
    buf.add(_labeled(neg, success=False))
    clear = _labeled(torch.tensor([[1.0, 1.0]]), success=True)
    ambiguous = _labeled(torch.tensor([[0.0, 0.0]]), success=True)
    assert compute_conf(clear, buf) > compute_conf(ambiguous, buf)


def test_conf_neutral_when_one_bucket_empty():
    buf = TrajectoryBuffer()
    buf.add(_labeled(torch.randn(3, 2), success=True))
    traj = _labeled(torch.randn(2, 2), success=True)
    assert compute_conf(traj, buf) == 0.5


def test_conf_neutral_when_both_buckets_empty():
    buf = TrajectoryBuffer()
    traj = _labeled(torch.randn(2, 2), success=True)
    assert compute_conf(traj, buf) == 0.5


def test_conf_threshold_default_when_buffer_small():
    buf = TrajectoryBuffer()
    assert conf_threshold(buf) == 0.5
    buf.add(_labeled(torch.randn(3, 2), success=True))
    assert conf_threshold(buf) == 0.5  # only 1 traj


def test_conf_threshold_within_unit_interval():
    torch.manual_seed(1)
    buf = TrajectoryBuffer()
    for _ in range(3):
        buf.add(_labeled(torch.randn(3, 2), success=True))
        buf.add(_labeled(torch.randn(3, 2), success=False))
    t = conf_threshold(buf)
    assert 0.0 <= t <= 1.0
