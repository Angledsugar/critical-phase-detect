"""Extract proprio sequences from LIBERO demo hdf5 → pickle for TLDR.

Reads /media/engineer/DATA/datasets/libero/<suite>/*_demo.hdf5 and emits a
list[np.ndarray] (one per demo, shape (T, 8)) where features are
[ee_pos(3), ee_ori(3), gripper_states(2)]. The pickle is consumed by
scripts/train_tldr.py via cfg.trajectories_path.

Run from main venv (.venv has h5py + numpy):
    .venv/bin/python scripts/extract_libero_proprio.py \\
        --suite libero_10 --out data/tldr_demos.pkl
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import h5py
import numpy as np


def extract_demo_proprio(demo_grp: h5py.Group) -> np.ndarray:
    obs = demo_grp["obs"]
    ee_pos = obs["ee_pos"][...]
    ee_ori = obs["ee_ori"][...]
    gripper = obs["gripper_states"][...]
    feat = np.concatenate([ee_pos, ee_ori, gripper], axis=-1).astype(np.float32)
    return feat  # (T, 8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_10")
    p.add_argument(
        "--root",
        default="/media/engineer/DATA/datasets/libero",
        help="LIBERO datasets root (suite subdir holds *_demo.hdf5)",
    )
    p.add_argument("--out", default="data/tldr_demos.pkl")
    args = p.parse_args()

    pattern = f"{args.root}/{args.suite}/*_demo.hdf5"
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no hdf5 found under {pattern}")
    print(f"[extract] suite={args.suite} files={len(files)}")

    trajs: list[np.ndarray] = []
    lengths: list[int] = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            data = f["data"]
            demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
            for k in demo_keys:
                arr = extract_demo_proprio(data[k])
                trajs.append(arr)
                lengths.append(arr.shape[0])
        print(f"[extract] {Path(fp).name}: {len(demo_keys)} demos")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(trajs, f)

    lens = np.asarray(lengths)
    print(
        f"[done] {len(trajs)} trajectories | "
        f"T: min={lens.min()} max={lens.max()} mean={lens.mean():.1f} | "
        f"state_dim={trajs[0].shape[1]} | "
        f"saved -> {out_path.resolve()} ({out_path.stat().st_size/1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
