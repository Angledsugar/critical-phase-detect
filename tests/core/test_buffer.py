"""TrajectoryBuffer — partition by success, all_latents concat, iteration."""
import pytest
import torch

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.trajectory import Trajectory


def _labeled(T: int, d: int, success: bool) -> Trajectory:
    return Trajectory(
        raw_states=list(range(T)),
        latents=torch.randn(T, d),
        actions=(),
        success=success,
    )


def test_partition_by_label():
    buf = TrajectoryBuffer()
    buf.add(_labeled(5, 4, True))
    buf.add(_labeled(7, 4, False))
    buf.add(_labeled(3, 4, True))
    assert buf.n_positive == 2
    assert buf.n_negative == 1
    assert len(buf) == 3


def test_rejects_unlabeled():
    buf = TrajectoryBuffer()
    unlabeled = Trajectory(
        raw_states=[0, 1, 2], latents=torch.randn(3, 4), actions=()
    )
    with pytest.raises(ValueError, match="unlabeled"):
        buf.add(unlabeled)


def test_all_latents_concat_shape():
    buf = TrajectoryBuffer()
    buf.add(_labeled(5, 4, True))
    buf.add(_labeled(7, 4, True))
    buf.add(_labeled(3, 4, False))

    pos = buf.all_latents(positive=True)
    neg = buf.all_latents(positive=False)
    assert pos.shape == (12, 4)
    assert neg.shape == (3, 4)


def test_all_latents_empty_bucket():
    buf = TrajectoryBuffer()
    out = buf.all_latents(positive=True)
    assert out.shape == (0, 0)


def test_extend_and_iter():
    buf = TrajectoryBuffer()
    trajs = [_labeled(5, 4, True), _labeled(5, 4, False), _labeled(5, 4, True)]
    buf.extend(trajs)
    assert len(list(buf)) == 3
