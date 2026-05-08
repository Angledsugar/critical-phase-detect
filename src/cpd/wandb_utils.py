"""wandb integration helpers.

All training / eval scripts log through these so wandb access is centralized.
Run group = experiment name from hydra config; full config is logged for
reproducibility (resolved + commit hash).
"""
from __future__ import annotations

from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


def init_run(cfg: DictConfig) -> Any:
    """Start a wandb run from a hydra config.

    cfg.wandb.project / .entity / .mode and cfg.experiment.name are read.
    Returns the wandb.Run handle.
    """
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity"),
        mode=cfg.wandb.get("mode", "online"),
        group=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    wandb.log(metrics, step=step)


def log_figure(fig: Any, name: str) -> None:
    """matplotlib Figure → wandb.Image."""
    wandb.log({name: wandb.Image(fig)})


def log_table(rows: list[list[Any]], columns: list[str], name: str) -> None:
    wandb.log({name: wandb.Table(columns=columns, data=rows)})


def finish() -> None:
    wandb.finish()
