"""Streamlit entrypoint for the RLT-style handoff annotator.

Usage:
    streamlit run scripts/run_annotator.py -- \
        --config configs/experiment/labeling.yaml --annotator-id alice
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cpd.labeling.annotator import run_annotator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment/labeling.yaml"),
        help="YAML with traj_dir / output_dir / rubric_path / pilot_set.",
    )
    parser.add_argument("--annotator-id", required=True)
    parser.add_argument("--traj-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rubric-path", type=Path, default=None)
    parser.add_argument("--pilot", action="store_true", help="Restrict to pilot_set.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = {}
    if args.config and args.config.exists():
        cfg = yaml.safe_load(args.config.read_text()) or {}

    traj_dir = args.traj_dir or Path(cfg.get("traj_dir", "data/trajectories"))
    output_dir = args.output_dir or Path(cfg.get("output_dir", "data/rlt_labels"))
    rubric_path = args.rubric_path or (
        Path(cfg["rubric_path"]) if cfg.get("rubric_path") else None
    )
    pilot_set = list(cfg.get("pilot_set", [])) if args.pilot else None

    run_annotator(
        traj_dir=traj_dir,
        output_dir=output_dir,
        annotator_id=args.annotator_id,
        rubric_path=rubric_path,
        pilot_set=pilot_set,
    )


if __name__ == "__main__":
    main()
