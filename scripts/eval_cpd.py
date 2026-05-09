"""Evaluation driver for CPD + RLT baseline. Paper §6.3.

Loads:
  cfg.eval.trajectories — torch.save'd list of cpd.core.Trajectory (or
                          dict with keys "trajectories" + optional "ids")
  cfg.eval.labels       — torch.save'd dict[traj_id, handoff_step]

Outputs:
  reports/runs/{date}_eval.csv — F1 metrics for CPD + RLT
  plots/handoff_{method}.png   — handoff-step distribution per method

Example invocation (one-liner):
  uv run python scripts/eval_cpd.py +eval.trajectories=data.pt \
      +eval.labels=labels.pt +eval.tolerance=2 hydra.run.dir=. \
      hydra.output_subdir=null
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from cpd.core.pipeline import DetectorPipeline
from cpd.eval.metrics import F1Metric
from cpd.eval.rlt_baseline import RLTBaseline
from cpd.viz.handoff import plot_handoff_distribution

logger = logging.getLogger(__name__)


def _load_trajectories(path: str | None) -> tuple[list, list]:
    """Load (trajectories, traj_ids) from `path` or return empty."""
    if not path:
        return [], []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trajectory file {p} not found.")
    payload = torch.load(p, weights_only=False)
    if isinstance(payload, dict) and "trajectories" in payload:
        trajs = list(payload["trajectories"])
        ids = list(payload.get("ids", range(len(trajs))))
    else:
        trajs = list(payload)
        ids = list(range(len(trajs)))
    return trajs, ids


def _load_labels(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Labels file {p} not found.")
    return torch.load(p, weights_only=False)


def _cpd_predict(pipeline: DetectorPipeline | None) -> dict:
    """Best-effort CPD predictions. T1 will fill core stubs; we tolerate
    NotImplementedError so this driver works as scaffolding today."""
    preds: dict = {}
    if pipeline is None:
        return preds
    try:
        pipeline.refresh_stats()
        raise RuntimeError(
            "Pipeline.refresh_stats returned without exception — "
            "CPD predictor not yet wired (T1)."
        )
    except NotImplementedError as e:
        logger.warning("CPD pipeline stubs incomplete (T1): %s", e)
    except Exception as e:  # noqa: BLE001
        logger.warning("CPD pipeline run failed: %s", e)
    return preds


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # WHY OmegaConf.select with default: `eval` may not be present in the base
    # config; users add it on the command line via `+eval.trajectories=...`.
    traj_path = OmegaConf.select(cfg, "eval.trajectories", default=None)
    labels_path = OmegaConf.select(cfg, "eval.labels", default=None)
    tolerance = int(OmegaConf.select(cfg, "eval.tolerance", default=3))
    rlt_epochs = int(OmegaConf.select(cfg, "eval.rlt_epochs", default=20))

    trajectories, traj_ids = _load_trajectories(traj_path)
    labels = _load_labels(labels_path)
    logger.info("Loaded %d trajectories and %d labels.", len(trajectories), len(labels))

    f1 = F1Metric(tolerance=tolerance)

    # CPD pipeline — scaffold for now; T1 will pass an instantiated pipeline.
    cpd_preds = _cpd_predict(None)

    # RLT baseline.
    rlt_preds: dict = {}
    if trajectories and labels:
        rlt = RLTBaseline()
        rlt.fit(trajectories, labels, traj_ids=traj_ids, epochs=rlt_epochs)
        rlt_preds = rlt.predict_all(trajectories, traj_ids=traj_ids)

    cpd_metrics = f1.compute(cpd_preds, labels) if labels else {}
    rlt_metrics = f1.compute(rlt_preds, labels) if labels else {}

    # CSV.
    today = datetime.now().strftime("%Y%m%d")
    reports_dir = Path("reports/runs")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"{today}_eval.csv"
    fieldnames = ["method", "f1", "precision", "recall", "mean_step_error", "tp", "fp", "fn"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in [("cpd", cpd_metrics), ("rlt", rlt_metrics)]:
            row: dict = {"method": method}
            for k in fieldnames[1:]:
                row[k] = metrics.get(k, float("nan")) if metrics else float("nan")
            writer.writerow(row)
    logger.info("Wrote %s", csv_path)

    # Plots.
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    if rlt_preds:
        fig = plot_handoff_distribution(rlt_preds, labels)
        fig.savefig(plots_dir / "handoff_rlt.png", dpi=150)
    if cpd_preds:
        fig = plot_handoff_distribution(cpd_preds, labels)
        fig.savefig(plots_dir / "handoff_cpd.png", dpi=150)
    logger.info("Plots saved to %s", plots_dir)


if __name__ == "__main__":
    main()
