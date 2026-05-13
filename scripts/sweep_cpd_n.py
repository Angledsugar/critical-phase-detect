"""Sweep number-of-rollouts N → trajectory-level F1 for has-critical vs is-failure.

For each task in --task-ids, for each N_per_task in --n-grid:
  1. take rollouts ep000..ep{N-1}
  2. LOO KDE: each rollout uses the other N-1 from same task as buffer
  3. compute r_t per step, find critical intervals (r_t < tau, min_run filter)
  4. label has_critical = (∃ interval)
Pool labels across tasks (per N), compute precision/recall/F1 vs is_failure.
Bootstrap CI across trajectories.

Outputs:
  reports/exp2/sweep/per_traj.json   # all traj records per N
  reports/exp2/sweep/f1_vs_n.json    # summary numbers
  reports/exp2/sweep/f1_vs_n.png     # plot
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.kde import compute_kde
from cpd.core.reward import Reward
from cpd.core.trajectory import Trajectory
from cpd.encoders.tldr import TLDREncoder


def encode_scaled(encoder: TLDREncoder, raw: np.ndarray, scale: float) -> torch.Tensor:
    with torch.no_grad():
        return encoder.encode(raw).detach() / scale


def find_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def trajectory_signals(latents: torch.Tensor, kde, *, tau: float, min_run: int) -> dict:
    reward = Reward(kde=kde)
    rs = reward.per_step(latents).cpu().numpy()
    mask = rs < tau
    intervals = [(s, e) for s, e in find_intervals(mask) if (e - s) >= min_run]
    n_crit = sum(e - s for s, e in intervals)
    longest = max((e - s for s, e in intervals), default=0)
    return dict(
        has_critical=bool(intervals),
        n_crit_steps=int(n_crit),
        critical_fraction=float(n_crit / max(1, len(rs))),
        longest_run=int(longest),
        sum_r=float(rs.sum()),
        min_r=float(rs.min()),
        mean_r=float(rs.mean()),
        T=int(len(rs)),
    )


def best_threshold_f1(scores: np.ndarray, is_failure: np.ndarray, *,
                      direction: str = "lower_is_fail") -> tuple[float, dict]:
    """Find threshold on `scores` that maximizes F1 vs is_failure label.
    direction='lower_is_fail': predict failure when score < threshold.
    """
    if scores.size == 0:
        return 0.0, dict(tp=0, fp=0, tn=0, fn=0, precision=0.0, recall=0.0, f1=0.0, accuracy=0.0)
    candidates = np.unique(scores)
    candidates = np.concatenate(([candidates[0] - 1.0], candidates, [candidates[-1] + 1.0]))
    best_f1 = -1.0; best_thr = 0.0; best_m = None
    for thr in candidates:
        pred = (scores < thr) if direction == "lower_is_fail" else (scores > thr)
        m = f1_from_labels(is_failure, pred)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; best_thr = float(thr); best_m = m
    return best_thr, best_m


def f1_from_labels(is_failure: np.ndarray, has_crit: np.ndarray) -> dict:
    tp = int(((is_failure) & (has_crit)).sum())
    fp = int(((~is_failure) & (has_crit)).sum())
    tn = int(((~is_failure) & (~has_crit)).sum())
    fn = int(((is_failure) & (~has_crit)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec, f1=f1, accuracy=acc)


def bootstrap_f1(is_failure: np.ndarray, has_crit: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(is_failure)
    if n == 0:
        return 0.0, 0.0
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = f1_from_labels(is_failure[idx], has_crit[idx])
        f1s.append(m["f1"])
    return float(np.quantile(f1s, 0.025)), float(np.quantile(f1s, 0.975))


def load_task_rollouts(rollout_root: Path, suite: str, task_id: int,
                       encoder, scale: float, max_n: int):
    task_dir = rollout_root / suite / f"task{task_id:02d}"
    eps = sorted(task_dir.glob("ep*.npz"))[:max_n]
    if not eps:
        return []
    out = []
    for fp in eps:
        npz = np.load(fp)
        latents = encode_scaled(encoder, npz["state"], scale)
        out.append({
            "path": fp,
            "task_id": task_id,
            "ep_id": fp.stem,
            "success": bool(npz["success"]),
            "T": int(npz["length"]),
            "latents": latents,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default="data/tldr_demos.pkl")
    p.add_argument("--encoder", default="checkpoints/tldr.pt")
    p.add_argument("--rollout-root", default="/media/engineer/DATA/datasets/cpd_rollouts/pi05_libero")
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--task-ids", type=int, nargs="+", default=[0])
    p.add_argument("--n-grid", type=int, nargs="+",
                   default=[10, 20, 30, 50, 70, 100, 140, 200])
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--min-run", type=int, default=3)
    p.add_argument("--out", default="reports/exp2/sweep")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = TLDREncoder.load(args.encoder)
    encoder.eval()
    with Path(args.demos).open("rb") as f:
        demo_raw: list[np.ndarray] = pickle.load(f)
    raw_finals = torch.stack([encoder.encode(r)[-1].detach() for r in demo_raw])
    scale = float(raw_finals.norm(dim=-1).mean())
    print(f"[sweep] latent scale = {scale:.3f}")

    rollout_root = Path(args.rollout_root)
    max_n = max(args.n_grid)

    task_rollouts: dict[int, list[dict]] = {}
    for tid in args.task_ids:
        rs = load_task_rollouts(rollout_root, args.suite, tid, encoder, scale, max_n)
        task_rollouts[tid] = rs
        succ = sum(r["success"] for r in rs)
        print(f"[sweep] task{tid:02d}: loaded {len(rs)} rollouts ({succ} succ / {len(rs)-succ} fail)")

    sweep_records: list[dict] = []
    summary_per_n: list[dict] = []

    for N in args.n_grid:
        is_fail_all, has_crit_all, sum_r_all = [], [], []
        per_task: list[dict] = []
        for tid in args.task_ids:
            rs = task_rollouts[tid][:N]
            if len(rs) < N:
                print(f"  [skip] task{tid:02d} only has {len(rs)} rollouts < N={N}")
                continue
            trajs = [
                Trajectory(
                    raw_states=[[] for _ in range(r["T"])],
                    latents=r["latents"],
                    success=r["success"],
                )
                for r in rs
            ]
            ifail_t, hc_t, sr_t = [], [], []
            for idx, (r, tr) in enumerate(zip(rs, trajs)):
                buf = TrajectoryBuffer()
                for j, other in enumerate(trajs):
                    if j == idx:
                        continue
                    buf.add(other)
                if buf.n_positive == 0 or buf.n_negative == 0:
                    sweep_records.append(dict(
                        N=N, task_id=tid, ep=r["ep_id"], success=r["success"],
                        has_critical=False, sum_r=0.0, degenerate=True,
                        n_pos=buf.n_positive, n_neg=buf.n_negative,
                    ))
                    ifail_t.append(not r["success"])
                    hc_t.append(False)
                    sr_t.append(0.0)
                    continue
                kde = compute_kde(buf)
                sig = trajectory_signals(tr.latents, kde, tau=args.tau, min_run=args.min_run)
                ifail_t.append(not r["success"])
                hc_t.append(sig["has_critical"])
                sr_t.append(sig["sum_r"])
                sweep_records.append(dict(
                    N=N, task_id=tid, ep=r["ep_id"], success=r["success"],
                    **sig, degenerate=False,
                    n_pos=buf.n_positive, n_neg=buf.n_negative,
                ))
            ifail_t = np.array(ifail_t)
            hc_t = np.array(hc_t)
            sr_t = np.array(sr_t)
            mt = f1_from_labels(ifail_t, hc_t)
            per_task.append(dict(task_id=tid, **mt))
            is_fail_all.append(ifail_t)
            has_crit_all.append(hc_t)
            sum_r_all.append(sr_t)

        if not is_fail_all:
            continue
        is_fail_all = np.concatenate(is_fail_all)
        has_crit_all = np.concatenate(has_crit_all)
        sum_r_all = np.concatenate(sum_r_all)
        m = f1_from_labels(is_fail_all, has_crit_all)
        lo, hi = bootstrap_f1(is_fail_all, has_crit_all, n_boot=args.n_boot, seed=args.seed)
        thr_best, m_sumr_best = best_threshold_f1(sum_r_all, is_fail_all, direction="lower_is_fail")
        m.update(dict(N=N, total_trajs=int(is_fail_all.size),
                      n_failures=int(is_fail_all.sum()),
                      n_successes=int((~is_fail_all).sum()),
                      f1_ci_lo=lo, f1_ci_hi=hi,
                      sumr_best_threshold=thr_best,
                      sumr_best_f1=m_sumr_best["f1"],
                      sumr_best_precision=m_sumr_best["precision"],
                      sumr_best_recall=m_sumr_best["recall"],
                      sumr_best_accuracy=m_sumr_best["accuracy"],
                      per_task=per_task))
        summary_per_n.append(m)
        print(f"  N={N:3d}  total={m['total_trajs']:3d} (succ={m['n_successes']:3d}/fail={m['n_failures']:3d})  "
              f"[has-crit] prec={m['precision']:.3f} rec={m['recall']:.3f} F1={m['f1']:.3f} CI=[{lo:.3f},{hi:.3f}]  "
              f"[sum_r<thr] best F1={m_sumr_best['f1']:.3f} thr={thr_best:+.2f}")

    (out_dir / "per_traj.json").write_text(json.dumps(sweep_records, indent=2))
    (out_dir / "f1_vs_n.json").write_text(json.dumps(summary_per_n, indent=2))

    # 95% threshold
    crossed = [s for s in summary_per_n if s["f1"] >= 0.95]
    threshold_n = crossed[0]["N"] if crossed else None
    print(f"[sweep] smallest N with F1 >= 0.95: {threshold_n}")

    if summary_per_n:
        ns = [s["N"] for s in summary_per_n]
        f1s = [s["f1"] for s in summary_per_n]
        los = [s["f1_ci_lo"] for s in summary_per_n]
        his = [s["f1_ci_hi"] for s in summary_per_n]
        precs = [s["precision"] for s in summary_per_n]
        recs = [s["recall"] for s in summary_per_n]
        f1s_sumr = [s["sumr_best_f1"] for s in summary_per_n]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ns, f1s, "-o", color="tab:blue", lw=2, label="F1 (has-critical, τ=0)")
        ax.fill_between(ns, los, his, color="tab:blue", alpha=0.18, label="95% CI bootstrap")
        ax.plot(ns, precs, "--", color="tab:green", lw=1.0, label="precision")
        ax.plot(ns, recs, "--", color="tab:red", lw=1.0, label="recall")
        ax.plot(ns, f1s_sumr, "-s", color="tab:orange", lw=1.6,
                label="F1 (sum_r < best-thr)")
        ax.axhline(0.95, color="k", lw=0.7, ls=":", label="0.95 threshold")
        if threshold_n is not None:
            ax.axvline(threshold_n, color="tab:purple", lw=0.7, ls="--",
                       label=f"first N≥0.95 (has-crit): N={threshold_n}")
        ax.set_xlabel("N rollouts (libero_10 task00, init_states cycled)")
        ax.set_ylabel("F1 (trajectory-level detection)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_title("CPD: F1 vs number of rollouts")
        fig.tight_layout()
        fig.savefig(out_dir / "f1_vs_n.png", dpi=140)
        plt.close(fig)
        print(f"[sweep] saved → {out_dir/'f1_vs_n.png'}")


if __name__ == "__main__":
    main()
