"""LIBERO env + demo source smoke tests. Skips cleanly if libero isn't installed."""
from __future__ import annotations

import os

import pytest
import torch

libero = pytest.importorskip("libero.libero")  # noqa: F401

from cpd.core.trajectory import Trajectory  # noqa: E402
from cpd.envs.libero import LiberoDemoSource, LiberoEnv  # noqa: E402

SUITE = "libero_long"
TASK_ID = 0


def _datasets_available() -> bool:
    try:
        from libero.libero import get_libero_path

        ds_root = get_libero_path("datasets")
    except Exception:
        return False
    return os.path.isdir(ds_root)


def test_to_trajectory_helper_zero_dim_latent():
    states = [{"x": 0}, {"x": 1}, {"x": 2}]
    actions = [[0.0] * 7, [0.0] * 7, [0.0] * 7]
    traj = LiberoDemoSource.to_trajectory(states, actions, success=True)
    assert isinstance(traj, Trajectory)
    assert len(traj) == 3
    assert traj.latents.shape == (3, 0)
    assert traj.latent_dim == 0
    assert traj.success is True


def test_env_reset_and_step():
    env = LiberoEnv(suite=SUITE, task_id=TASK_ID, image_size=64)
    try:
        obs = env.reset()
        assert isinstance(obs, dict)
        action = [0.0] * 7
        next_obs, reward, done, info = env.step(action)
        assert isinstance(next_obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert env.task_id.startswith("libero_10/")
    finally:
        env.close()


def test_demos_loads_at_least_one_trajectory():
    if not _datasets_available():
        pytest.skip("LIBERO datasets not downloaded.")
    src = LiberoDemoSource(suite=SUITE, task_id=TASK_ID, max_demos=1)
    trajs = src.demos(task_id=None, n=1)
    assert len(trajs) >= 1
    traj = trajs[0]
    assert traj.success is True
    assert len(traj) > 0
    assert traj.latents.ndim == 2
    assert traj.latents.shape[1] == 0
    assert isinstance(traj.latents, torch.Tensor)
