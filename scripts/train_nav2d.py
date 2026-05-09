"""Smoke driver for T1 — Nav2D end-to-end self-supervised CPD pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.labeler import G2Labeler
from cpd.core.pipeline import DetectorPipeline
from cpd.core.trajectory import Trajectory
from cpd.encoders.identity import IdentityEncoder
from cpd.envs.nav2d import Nav2DDemoSource, Nav2DEnv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = str(_REPO_ROOT / "configs")


def _random_rollouts(env: Nav2DEnv, n: int, seed: int) -> list[Trajectory]:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    out: list[Trajectory] = []
    for _ in range(n):
        start = (torch.rand(2, generator=gen) * float(env.grid_size)).tolist()
        obs = env.reset(start=tuple(start))
        states: list[torch.Tensor] = [obs.clone()]
        done = False
        while not done:
            action = (torch.rand(2, generator=gen) * 2.0 - 1.0) * env.step_size
            obs, _r, done, _info = env.step(action)
            states.append(obs.clone())
        latents = torch.stack(states, dim=0).to(torch.float32)
        out.append(
            Trajectory(
                raw_states=[s.tolist() for s in states],
                latents=latents,
                actions=(),
            )
        )
    return out


@hydra.main(config_path=_CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    exp_seed = int(cfg.experiment.get("seed", 0))
    n_demos = int(cfg.experiment.get("n_demos", 20))
    n_rollouts = int(cfg.experiment.get("n_rollouts", 50))
    quantile = float(cfg.experiment.get("quantile", 0.95))
    use_wandb = bool(cfg.wandb.get("enabled", False))

    torch.manual_seed(exp_seed)

    env = Nav2DEnv(seed=exp_seed)
    encoder = IdentityEncoder(latent_dim=2)
    assert encoder.latent_dim == 2

    demo_source = Nav2DDemoSource(env=env, seed=exp_seed)
    demos = demo_source.demos(task_id="nav2d", n=n_demos)
    assert all(d.success is True for d in demos), "Greedy demos must succeed."

    labeler = G2Labeler.from_demos(demos, quantile=quantile)
    buffer = TrajectoryBuffer()
    buffer.extend(demos)
    pipeline = DetectorPipeline(labeler=labeler, buffer=buffer)

    rollouts = _random_rollouts(env, n=n_rollouts, seed=exp_seed + 1)
    pipeline.ingest(rollouts)

    n_pos = pipeline.buffer.n_positive
    n_neg = pipeline.buffer.n_negative
    h = pipeline.kde.h if pipeline.kde is not None else float("nan")

    held_out = rollouts[-1] if rollouts else demos[-1]
    sample_reward = pipeline.reward.trajectory(held_out) if pipeline.reward else float("nan")

    print(f"[nav2d] |B+| = {n_pos}, |B-| = {n_neg}")
    print(f"[nav2d] KDE bandwidth h = {h:.6f}")
    print(f"[nav2d] sample trajectory reward = {sample_reward:.6f}")

    if use_wandb:
        from cpd import wandb_utils  # local import — only when needed

        wandb_utils.init_run(cfg)
        wandb_utils.log_metrics(
            {
                "buffer/n_pos": float(n_pos),
                "buffer/n_neg": float(n_neg),
                "kde/h": float(h),
                "reward/sample_trajectory": float(sample_reward),
            }
        )
        wandb_utils.finish()

    # Suppress hydra's internal cfg from leaking — return clean to CLI.
    _ = OmegaConf.to_container(cfg, resolve=True)


if __name__ == "__main__":
    sys.exit(main())
