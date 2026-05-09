"""Nav2DEnv + Nav2DDemoSource — reset/step contract, demos all succeed."""
import torch

from cpd.envs.nav2d import Nav2DDemoSource, Nav2DEnv


def test_reset_returns_position_in_bounds():
    env = Nav2DEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (2,)
    assert torch.all(obs >= 0.0)
    assert torch.all(obs <= env.grid_size)


def test_step_clips_action_to_step_size():
    env = Nav2DEnv(seed=0)
    obs = env.reset(start=(0.5, 0.5))
    huge = torch.tensor([10.0, 10.0])
    new_obs, _r, _done, _info = env.step(huge)
    delta = (new_obs - obs).abs().max().item()
    assert delta <= env.step_size + 1e-6


def test_step_terminates_on_max_steps():
    env = Nav2DEnv(max_steps=3, seed=0)
    env.reset(start=(0.0, 0.0))
    done = False
    for _ in range(3):
        _o, _r, done, _i = env.step(torch.zeros(2))
    assert done is True


def test_step_terminates_on_success():
    env = Nav2DEnv(seed=0, success_radius=0.1)
    env.reset(start=(0.85, 0.85))  # within success radius of (0.9, 0.9)
    _o, _r, done, info = env.step(torch.zeros(2))
    assert done is True
    assert info["success"] is True


def test_demos_all_succeed_and_within_max_steps():
    env = Nav2DEnv(seed=0, max_steps=50)
    src = Nav2DDemoSource(env=env, seed=0)
    demos = src.demos(task_id="nav2d", n=10)
    assert len(demos) == 10
    for d in demos:
        assert d.success is True
        assert len(d) <= env.max_steps + 1  # +1 for initial state
        assert d.latents.shape[1] == 2


def test_demo_latents_match_states():
    env = Nav2DEnv(seed=1, max_steps=50)
    src = Nav2DDemoSource(env=env, seed=1)
    demos = src.demos(task_id="nav2d", n=2)
    for d in demos:
        assert d.latents.shape[0] == len(d.raw_states)
