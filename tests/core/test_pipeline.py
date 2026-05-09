"""DetectorPipeline — ingest+refresh, idempotent refresh, both-bucket trigger."""
import torch

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.kde import KDEStats
from cpd.core.labeler import G2Labeler
from cpd.core.pipeline import DetectorPipeline
from cpd.core.reward import Reward
from cpd.core.trajectory import Trajectory


def _traj(final: torch.Tensor, T: int = 4) -> Trajectory:
    d = final.shape[0]
    latents = torch.zeros(T, d)
    latents[-1] = final
    return Trajectory(raw_states=list(range(T)), latents=latents, actions=())


def _make_pipeline() -> DetectorPipeline:
    finals = torch.tensor([[0.0, 0.0], [0.05, 0.0], [0.0, 0.05], [0.05, 0.05]])
    demos = [_traj(f) for f in finals]
    labeler = G2Labeler.from_demos(demos, quantile=0.95)
    buf = TrajectoryBuffer()
    return DetectorPipeline(labeler=labeler, buffer=buf)


def test_ingest_labels_and_buffers():
    pipe = _make_pipeline()
    near = _traj(torch.tensor([0.0, 0.0]))
    far = _traj(torch.tensor([10.0, 10.0]))
    pipe.ingest([near, far])
    assert pipe.buffer.n_positive == 1
    assert pipe.buffer.n_negative == 1


def test_ingest_triggers_refresh_when_both_buckets_nonempty():
    pipe = _make_pipeline()
    near = _traj(torch.tensor([0.0, 0.0]))
    pipe.ingest(near)
    assert pipe.kde is None and pipe.reward is None  # only positives so far
    far = _traj(torch.tensor([10.0, 10.0]))
    pipe.ingest(far)
    assert isinstance(pipe.kde, KDEStats)
    assert isinstance(pipe.reward, Reward)


def test_refresh_idempotent():
    pipe = _make_pipeline()
    pipe.ingest(_traj(torch.tensor([0.0, 0.0])))
    pipe.ingest(_traj(torch.tensor([10.0, 10.0])))
    h1 = pipe.kde.h
    pipe.refresh_stats()
    h2 = pipe.kde.h
    pipe.refresh_stats()
    h3 = pipe.kde.h
    assert h1 == h2 == h3


def test_ingest_accepts_pre_labeled():
    pipe = _make_pipeline()
    t = _traj(torch.tensor([10.0, 10.0])).with_label(success=True)
    pipe.ingest(t)
    assert pipe.buffer.n_positive == 1  # respects existing label


def test_refresh_noop_on_empty_buffer():
    pipe = _make_pipeline()
    pipe.refresh_stats()
    assert pipe.kde is None
    assert pipe.reward is None
