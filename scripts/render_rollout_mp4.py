"""Render pi05 rollout npz files → side-by-side mp4 (agentview + wrist).

Usage:
    .venv/bin/python scripts/render_rollout_mp4.py
    .venv/bin/python scripts/render_rollout_mp4.py \\
        --in /media/engineer/DATA/datasets/cpd_rollouts/pi05_libero/libero_10/task00 \\
        --out reports/exp1/videos --fps 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def annotate_frame(
    agent: np.ndarray, wrist: np.ndarray, *, step: int, total: int, success: bool, label: str
) -> np.ndarray:
    side = np.concatenate([agent, wrist], axis=1)  # (H, 2W, 3) uint8
    h, w = side.shape[:2]
    cv2.putText(side, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(side, f"step {step:03d}/{total:03d}", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    color = (0, 200, 0) if success else (0, 60, 220)
    tag = "SUCCESS" if success else "FAILURE"
    cv2.putText(side, tag, (w - 130, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return side


def render_episode(npz_path: Path, out_path: Path, *, fps: int) -> dict:
    npz = np.load(npz_path)
    agent = npz["agentview"]
    wrist = npz["wrist"]
    success = bool(npz["success"])
    T = int(npz["length"])
    label = str(npz["task_description"]) if "task_description" in npz.files else npz_path.stem

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    try:
        for t in range(T):
            frame = annotate_frame(
                agent[t], wrist[t], step=t, total=T, success=success, label=label
            )
            writer.append_data(frame)
    finally:
        writer.close()
    return {"path": out_path, "T": T, "success": success}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in", dest="in_dir",
        default="/media/engineer/DATA/datasets/cpd_rollouts/pi05_libero/libero_10/task00",
    )
    p.add_argument("--out", default="reports/exp1/videos")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = sorted(in_dir.glob("ep*.npz"))
    if not eps:
        raise SystemExit(f"no ep*.npz under {in_dir}")
    print(f"[render] {len(eps)} episodes from {in_dir}")
    print(f"[render] writing → {out_dir} (fps={args.fps})")

    for npz_path in eps:
        tag = "succ" if bool(np.load(npz_path)["success"]) else "FAIL"
        out_path = out_dir / f"{npz_path.stem}_{tag}.mp4"
        info = render_episode(npz_path, out_path, fps=args.fps)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {npz_path.name} → {out_path.name}  T={info['T']:3d}  {size_mb:5.1f} MB")


if __name__ == "__main__":
    main()
