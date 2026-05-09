"""Evaluation metrics for CPD. Paper §6.3.

F1Metric — handoff-step matching with ± tolerance window.
DownstreamSuccessMetric — success rate over n rollout episodes.

Both implement the Metric protocol from cpd.eval.base. Predictions are
per-trajectory integer handoff steps; ground-truth is the matching dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class F1Metric:
    """F1 of CPD's predicted handoff step vs RLT ground-truth label.

    Per-trajectory binary correctness: pred is "correct" iff |pred - gt| <= tolerance.
    Then standard precision / recall / F1 aggregated over the dataset.

    Predictions / ground-truth are dict[traj_id, step_index]. A traj may be
    absent from `predictions` (no detection) or from `ground_truth` (no label),
    which translates to FN or FP respectively.
    """

    tolerance: int = 3
    name: str = "f1"

    def compute(
        self,
        predictions: dict[Any, int | None],
        ground_truth: dict[Any, int],
    ) -> dict[str, float]:
        # WHY treat None / missing as no-detection: a baseline may abstain on
        # some trajectories; we still want to count that as a missed detection.
        tp = 0
        fp = 0
        fn = 0

        for tid, gt_step in ground_truth.items():
            pred_step = predictions.get(tid)
            if pred_step is None:
                fn += 1
                continue
            if abs(int(pred_step) - int(gt_step)) <= self.tolerance:
                tp += 1
            else:
                # Wrong step counts as both a false positive (we predicted an
                # incorrect handoff) and a false negative (we missed the true one).
                fp += 1
                fn += 1

        # Predictions for trajectories with no ground truth → false positives.
        for tid, pred_step in predictions.items():
            if pred_step is None:
                continue
            if tid not in ground_truth:
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Mean step error over matched (i.e., commonly-keyed) trajectories.
        common = [
            (predictions[t], ground_truth[t])
            for t in ground_truth
            if predictions.get(t) is not None
        ]
        mean_step_error = (
            sum(abs(int(p) - int(g)) for p, g in common) / len(common)
            if common
            else 0.0
        )

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "mean_step_error": mean_step_error,
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }


@dataclass
class DownstreamSuccessMetric:
    """Task success rate of `policy` rolled out in `env` for n_episodes.

    `env` must follow the Env protocol (reset() -> obs; step(a) -> (obs, r, done, info)).
    `policy` must follow the Policy protocol (predict(obs) -> action).
    Episode success is read from the final `info["success"]`; if absent we
    fall back to `final_reward > 0`.
    """

    env: Any
    policy: Any
    n_episodes: int = 20
    max_steps: int = 1000
    name: str = "success_rate"

    def compute(
        self,
        predictions: Any = None,
        ground_truth: Any = None,
    ) -> dict[str, float]:
        # predictions / ground_truth are unused — kept for Metric protocol parity.
        del predictions, ground_truth
        successes = 0
        for _ in range(self.n_episodes):
            obs = self.env.reset()
            done = False
            steps = 0
            last_info: dict = {}
            last_reward = 0.0
            while not done and steps < self.max_steps:
                action = self.policy.predict(obs)
                obs, reward, done, info = self.env.step(action)
                last_info = info if info is not None else {}
                last_reward = reward
                steps += 1
            if "success" in last_info:
                if bool(last_info["success"]):
                    successes += 1
            elif last_reward > 0:
                successes += 1
        return {
            "success_rate": successes / self.n_episodes,
            "n_episodes": float(self.n_episodes),
            "n_successes": float(successes),
        }
