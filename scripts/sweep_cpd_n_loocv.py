"""LOO-CV F1 vs N — honest detection F1 across trajectory-level rules.

Loads scripts/sweep_cpd_n.py's per_traj.json (already has the trajectory-level
signals per (N, episode)) and computes leave-one-out cross-validated F1 for
three classification rules:

  - has_critical (any 3-step run with r_t<0)   ← original loose rule
  - longest_run >= L_thr                       ← duration of worst dip
  - n_crit_steps >= K_thr                      ← total critical-step count
  - critical_fraction >= F_thr                 ← critical fraction of T

For each (N, rule) the threshold is chosen by best-F1 on the N-1 training
trajectories, then applied to the held-out one. Aggregating predictions over
all 200 LOO folds gives a single F1 that does not see its own answer.

Output:
  reports/exp2/sweep/loocv_f1_vs_n.json
  reports/exp2/sweep/loocv_f1_vs_n.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def best_thr(scores: np.ndarray, is_fail: np.ndarray) -> tuple[float, float]:
    """Pick threshold maximizing F1 with rule (scores >= thr -> failure).

    Returns (best_thr, best_f1). If is_fail has only one class, returns +inf,0.
    """
    if len(np.unique(is_fail)) < 2:
        return float(np.inf), 0.0
    cands = np.unique(scores)
    best = (-1.0, float(cands[0]))
    for t in cands:
        pred = scores >= t
        tp = int((pred & is_fail).sum())
        fp = int((pred & ~is_fail).sum())
        fn = int((~pred & is_fail).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], best[0]


def loo_predict(scores: np.ndarray, is_fail: np.ndarray) -> dict:
    """LOO-CV: for each i pick threshold on N-1 others, predict on i.

    Returns aggregated metrics over the full LOO prediction vector.
    """
    n = len(scores)
    preds = np.zeros(n, dtype=bool)
    thrs = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.arange(n) != i
        t, _ = best_thr(scores[mask], is_fail[mask])
        thrs[i] = t
        preds[i] = scores[i] >= t
    tp = int((preds & is_fail).sum())
    fp = int((preds & ~is_fail).sum())
    fn = int((~preds & is_fail).sum())
    tn = int((~preds & ~is_fail).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(
        f1=f1, prec=prec, rec=rec, tp=tp, fp=fp, fn=fn, tn=tn,
        thr_median=float(np.median(thrs)), thr_iqr=[float(np.percentile(thrs, 25)),
                                                    float(np.percentile(thrs, 75))],
    )


def bootstrap_f1_ci(
    scores: np.ndarray, is_fail: np.ndarray, *, n_boot: int = 1000, seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap LOO F1 confidence interval (resampling episodes)."""
    rng = np.random.default_rng(seed)
    n = len(scores)
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s_b, f_b = scores[idx], is_fail[idx]
        if len(np.unique(f_b)) < 2:
            continue
        r = loo_predict(s_b, f_b)
        f1s.append(r["f1"])
    if not f1s:
        return 0.0, 0.0
    return float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-traj", default="reports/exp2/sweep/per_traj.json")
    p.add_argument("--out-dir", default="reports/exp2/sweep")
    p.add_argument("--n-boot", type=int, default=200,
                   help="bootstrap iters for CI (LOO inside each is O(N²), keep small)")
    args = p.parse_args()

    data = json.load(open(args.per_traj))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = [
        ("has_critical", lambda r: float(r["has_critical"])),
        ("longest_run", lambda r: float(r["longest_run"])),
        ("n_crit_steps", lambda r: float(r["n_crit_steps"])),
        ("critical_fraction", lambda r: float(r["critical_fraction"])),
    ]

    ns = sorted({r["N"] for r in data})
    results = {name: {} for name, _ in rules}

    print(f"{'N':>4} | " + " | ".join(f"{nm:>18s}" for nm, _ in rules))
    print("-" * (6 + 21 * len(rules)))
    for N in ns:
        rows = [r for r in data if r["N"] == N]
        is_fail = np.array([not r["success"] for r in rows])
        row_strs = []
        for name, getter in rules:
            scores = np.array([getter(r) for r in rows], dtype=float)
            res = loo_predict(scores, is_fail)
            lo, hi = bootstrap_f1_ci(scores, is_fail, n_boot=args.n_boot, seed=N * 7)
            res["ci_lo"] = lo
            res["ci_hi"] = hi
            results[name][N] = res
            row_strs.append(f"F1={res['f1']:.3f} [{lo:.2f},{hi:.2f}]")
        print(f"{N:4d} | " + " | ".join(f"{s:>18s}" for s in row_strs))

    (out_dir / "loocv_f1_vs_n.json").write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = {"has_critical": "tab:gray", "longest_run": "tab:blue",
               "n_crit_steps": "tab:red", "critical_fraction": "tab:green"}
    labels = {
        "has_critical": "has-critical (τ=0, min-run=3) — original loose rule",
        "longest_run": "longest critical run ≥ L*",
        "n_crit_steps": "total critical steps ≥ K*  (best rule)",
        "critical_fraction": "critical fraction ≥ F*",
    }
    for name, _ in rules:
        ns_list = sorted(results[name].keys())
        f1s = [results[name][n]["f1"] for n in ns_list]
        lo = [results[name][n]["ci_lo"] for n in ns_list]
        hi = [results[name][n]["ci_hi"] for n in ns_list]
        c = palette[name]
        ax.fill_between(ns_list, lo, hi, color=c, alpha=0.12)
        ax.plot(ns_list, f1s, "-o", color=c, lw=1.6, label=labels[name])
    ax.axhline(0.95, color="k", lw=0.7, ls="--", alpha=0.5, label="F1 = 0.95 target")
    ax.set_xlabel("Number of rollouts (N)")
    ax.set_ylabel("Detection F1 (LOO-CV, has-critical-phase vs failure)")
    ax.set_title("CPD detection F1 vs N — task00 (LIBERO-Long, π_0.5)\n"
                 "187 succ / 13 fail of 200 episodes — bootstrap 95% CI shown as band")
    ax.set_xticks(ns_list)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "loocv_f1_vs_n.png", dpi=140)
    plt.close(fig)
    print(f"\n→ {out_dir/'loocv_f1_vs_n.png'}")
    print(f"→ {out_dir/'loocv_f1_vs_n.json'}")


if __name__ == "__main__":
    main()
