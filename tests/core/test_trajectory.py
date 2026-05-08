"""Trajectory dataclass — validation, properties, with_label."""
import pytest
import torch

from cpd.core.trajectory import Trajectory


def _make(T: int = 5, d: int = 4, *, with_actions: bool = True) -> Trajectory:
    return Trajectory(
        raw_states=list(range(T)),
        latents=torch.randn(T, d),
        actions=tuple(range(T)) if with_actions else (),
    )


def test_basic_construction():
    traj = _make(T=10, d=4)
    assert len(traj) == 10
    assert traj.latent_dim == 4
    assert traj.final_latent.shape == (4,)
    assert not traj.labeled
    assert traj.success is None
    assert traj.conf is None


def test_no_actions_allowed():
    traj = _make(T=5, d=4, with_actions=False)
    assert len(traj) == 5
    assert len(traj.actions) == 0


def test_validates_latent_rank():
    with pytest.raises(ValueError, match="2-D|T, d"):
        Trajectory(raw_states=[0], latents=torch.randn(4), actions=())


def test_validates_state_length():
    with pytest.raises(ValueError, match="raw_states"):
        Trajectory(raw_states=[0, 1, 2], latents=torch.randn(5, 4), actions=())


def test_validates_action_length():
    with pytest.raises(ValueError, match="actions"):
        Trajectory(raw_states=list(range(5)), latents=torch.randn(5, 4), actions=(0, 1))


def test_with_label_returns_copy():
    traj = _make()
    labeled = traj.with_label(success=True, conf=0.9)
    assert labeled.success is True
    assert labeled.conf == 0.9
    assert labeled.labeled
    # original unchanged
    assert traj.success is None
    assert not traj.labeled
    # latents shared (not deep-copied — fine since frozen)
    assert labeled.latents is traj.latents


def test_metadata_default_empty():
    traj = _make()
    assert traj.metadata == {}
