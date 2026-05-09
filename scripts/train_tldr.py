"""Train the TLDR encoder via triplet contrastive loss. Paper §3.3.

Hydra entry point. Defaults to the synthetic 1-D-time toy: trajectories are
random walks in state_dim space with monotone drift, so step distance is the
dominant temporal signal — a sanity smoke for the triplet objective.

A real-demo path can be supplied through `trajectories_path` (pickled
list[Trajectory] or list of dicts/arrays with `.raw_states`).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from cpd.encoders.tldr import TLDREncoder
from cpd.encoders.tldr_train import TLDRTrainer


def _make_synthetic_trajectories(
    *,
    num_trajectories: int,
    traj_length: int,
    state_dim: int,
    drift: float,
    noise: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Random-walk trajectories with drift along a per-traj direction.

    s_{t+1} = s_t + drift * direction + noise * eps. Step distance therefore
    correlates strongly with state distance — exactly what TLDR should
    discover. Encoded as dicts with `.raw_states` attribute via SimpleTraj.
    """
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(num_trajectories):
        direction = rng.standard_normal(state_dim)
        direction /= np.linalg.norm(direction) + 1e-8
        s0 = rng.standard_normal(state_dim) * 0.5
        states = np.empty((traj_length, state_dim), dtype=np.float32)
        states[0] = s0
        for t in range(1, traj_length):
            eps = rng.standard_normal(state_dim) * noise
            states[t] = states[t - 1] + drift * direction + eps
        trajs.append(_SimpleTraj(states))
    return trajs


class _SimpleTraj:
    """Minimal duck-typed trajectory for the trainer (just `.raw_states`)."""

    def __init__(self, raw_states: np.ndarray) -> None:
        self.raw_states = raw_states


def _load_trajectories(path: str | Path) -> list[Any]:
    path = Path(path)
    with path.open("rb") as f:
        trajs = pickle.load(f)
    if not isinstance(trajs, list):
        raise ValueError(f"{path}: expected a list of trajectories, got {type(trajs)}")
    return trajs


@hydra.main(config_path="../configs/encoder", config_name="tldr", version_base=None)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.trajectories_path:
        trajectories = _load_trajectories(cfg.trajectories_path)
        print(f"[tldr] loaded {len(trajectories)} trajectories from {cfg.trajectories_path}")
    else:
        trajectories = _make_synthetic_trajectories(
            num_trajectories=cfg.synthetic.num_trajectories,
            traj_length=cfg.synthetic.traj_length,
            state_dim=cfg.state_dim,
            drift=cfg.synthetic.drift,
            noise=cfg.synthetic.noise,
            seed=cfg.seed,
        )
        print(f"[tldr] synthetic: {len(trajectories)} trajs × {cfg.synthetic.traj_length} steps")

    encoder = TLDREncoder(
        state_dim=cfg.state_dim,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    )
    trainer = TLDRTrainer(
        encoder,
        lr=cfg.lr,
        k_pos=cfg.k_pos,
        K_neg=cfg.K_neg,
        margin=cfg.margin,
        batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
    )

    wandb_run = None
    if cfg.wandb.enabled:
        from cpd import wandb_utils

        wandb_run = wandb_utils.init_run(cfg)

    print(f"[tldr] training {cfg.epochs} epochs × {cfg.num_batches_per_epoch} batches")
    for epoch in range(cfg.epochs):
        metrics = trainer.train_epoch(
            trajectories,
            num_batches=cfg.num_batches_per_epoch,
            seed=cfg.seed + epoch,
        )
        print(
            f"[tldr] epoch {epoch + 1:3d}/{cfg.epochs}: "
            f"loss={metrics['loss']:.4f} viol={metrics['violation_rate']:.3f} "
            f"d_ap={metrics['d_ap']:.3f} d_an={metrics['d_an']:.3f}"
        )
        if wandb_run is not None:
            from cpd import wandb_utils

            wandb_utils.log_metrics({f"train/{k}": v for k, v in metrics.items()}, step=epoch)

    weights_path = Path(cfg.weights_path)
    encoder.save(weights_path)
    print(f"[tldr] saved weights → {weights_path.resolve()}")

    if wandb_run is not None:
        from cpd import wandb_utils

        wandb_utils.finish()


if __name__ == "__main__":
    main()
