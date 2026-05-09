"""Handoff-step distribution + scatter. Paper Fig 4 companion."""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_handoff_distribution(
    predictions: dict[Any, int],
    ground_truth: dict[Any, int],
) -> Figure:
    """Two panels: (left) histograms of pred vs gt step indices,
    (right) scatter of pred vs gt with y=x reference."""
    pred_vals = [int(v) for v in predictions.values() if v is not None]
    gt_vals = [int(v) for v in ground_truth.values()]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: histograms.
    ax = axes[0]
    if pred_vals or gt_vals:
        all_vals = pred_vals + gt_vals
        lo, hi = min(all_vals), max(all_vals)
        # WHY +2 buckets: matplotlib expects bin edges, want a tail bin so
        # the rightmost value is visible.
        bins = max(10, hi - lo + 2)
        if pred_vals:
            ax.hist(pred_vals, bins=bins, alpha=0.55, color="C0", label="pred")
        if gt_vals:
            ax.hist(gt_vals, bins=bins, alpha=0.55, color="C3", label="ground truth")
    ax.set_xlabel("handoff step")
    ax.set_ylabel("count")
    ax.set_title("Handoff-step histograms")
    if pred_vals or gt_vals:
        ax.legend(loc="best")

    # Right: scatter pred vs gt + y=x.
    ax = axes[1]
    paired_pred: list[int] = []
    paired_gt: list[int] = []
    for tid, gt in ground_truth.items():
        p = predictions.get(tid)
        if p is None:
            continue
        paired_pred.append(int(p))
        paired_gt.append(int(gt))
    if paired_pred:
        ax.scatter(paired_gt, paired_pred, color="C0", s=18, alpha=0.7)
        lo = min(min(paired_gt), min(paired_pred))
        hi = max(max(paired_gt), max(paired_pred))
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("ground-truth handoff step")
    ax.set_ylabel("predicted handoff step")
    ax.set_title("Predicted vs ground-truth")

    fig.tight_layout()
    return fig


__all__ = ["plot_handoff_distribution"]
