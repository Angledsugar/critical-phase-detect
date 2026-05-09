"""F1Metric + DownstreamSuccessMetric tests."""
from __future__ import annotations

from cpd.eval.metrics import DownstreamSuccessMetric, F1Metric


def test_f1_perfect_match():
    f1 = F1Metric(tolerance=0)
    preds = {0: 5, 1: 10, 2: 15}
    gt = {0: 5, 1: 10, 2: 15}
    out = f1.compute(preds, gt)
    assert out["f1"] == 1.0
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["mean_step_error"] == 0.0


def test_f1_within_tolerance():
    f1 = F1Metric(tolerance=3)
    preds = {0: 7, 1: 12, 2: 14}  # off by 2, 2, 1 — all within tolerance
    gt = {0: 5, 1: 10, 2: 15}
    out = f1.compute(preds, gt)
    assert out["f1"] == 1.0
    assert abs(out["mean_step_error"] - (2 + 2 + 1) / 3) < 1e-9


def test_f1_no_detection():
    f1 = F1Metric(tolerance=2)
    preds: dict[int, int | None] = {0: None, 1: None, 2: None}
    gt = {0: 5, 1: 10, 2: 15}
    out = f1.compute(preds, gt)
    assert out["f1"] == 0.0
    assert out["recall"] == 0.0
    assert out["tp"] == 0.0
    assert out["fn"] == 3.0


def test_f1_all_wrong():
    f1 = F1Metric(tolerance=1)
    preds = {0: 50, 1: 100, 2: 150}  # way off
    gt = {0: 5, 1: 10, 2: 15}
    out = f1.compute(preds, gt)
    assert out["f1"] == 0.0
    assert out["tp"] == 0.0
    # Each wrong prediction counts as both FP and FN.
    assert out["fp"] == 3.0
    assert out["fn"] == 3.0


def test_f1_partial():
    f1 = F1Metric(tolerance=1)
    preds = {0: 5, 1: 100, 2: 16}  # correct, wrong, correct
    gt = {0: 5, 1: 10, 2: 15}
    out = f1.compute(preds, gt)
    assert out["tp"] == 2.0
    assert out["fp"] == 1.0
    assert out["fn"] == 1.0
    # precision = recall = 2/3, F1 = 2/3.
    assert abs(out["precision"] - 2 / 3) < 1e-9
    assert abs(out["recall"] - 2 / 3) < 1e-9
    assert abs(out["f1"] - 2 / 3) < 1e-9


def test_f1_extra_predictions():
    f1 = F1Metric(tolerance=1)
    preds = {0: 5, 1: 10, 2: 15, 99: 7}  # last one has no gt
    gt = {0: 5, 1: 10, 2: 15}
    out = f1.compute(preds, gt)
    # Three correct + one extra FP.
    assert out["tp"] == 3.0
    assert out["fp"] == 1.0
    assert out["fn"] == 0.0


def test_f1_empty():
    f1 = F1Metric()
    out = f1.compute({}, {})
    # No data → all zero, no exception.
    assert out["f1"] == 0.0
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0


def test_f1_metric_protocol():
    f1 = F1Metric()
    assert f1.name == "f1"
    assert callable(f1.compute)


# --- DownstreamSuccessMetric tests --- #


class _StubEnv:
    """Episodic env whose every episode succeeds with probability `success_prob`.

    Cycles through a deterministic pattern so the success rate is predictable
    in tests without RNG seeding pain.
    """

    def __init__(self, success_pattern: list[bool]) -> None:
        self.success_pattern = success_pattern
        self._idx = -1
        self.task_id = "stub"

    def reset(self):
        self._idx += 1
        return 0

    def step(self, action):
        # del action  # unused
        success = self.success_pattern[self._idx % len(self.success_pattern)]
        # Episode terminates immediately, with reward = 1 on success.
        return 0, (1.0 if success else 0.0), True, {"success": success}


class _StubPolicy:
    def predict(self, obs):
        del obs
        return 0

    def refine_step(self, batch):
        del batch
        return {}


def test_downstream_success_all_success():
    env = _StubEnv(success_pattern=[True])
    policy = _StubPolicy()
    metric = DownstreamSuccessMetric(env=env, policy=policy, n_episodes=10)
    out = metric.compute()
    assert out["success_rate"] == 1.0
    assert out["n_successes"] == 10.0


def test_downstream_success_all_fail():
    env = _StubEnv(success_pattern=[False])
    policy = _StubPolicy()
    metric = DownstreamSuccessMetric(env=env, policy=policy, n_episodes=10)
    out = metric.compute()
    assert out["success_rate"] == 0.0
    assert out["n_successes"] == 0.0


def test_downstream_success_half():
    env = _StubEnv(success_pattern=[True, False])
    policy = _StubPolicy()
    metric = DownstreamSuccessMetric(env=env, policy=policy, n_episodes=20)
    out = metric.compute()
    assert out["success_rate"] == 0.5
    assert out["n_successes"] == 10.0


def test_downstream_success_protocol():
    env = _StubEnv(success_pattern=[True])
    policy = _StubPolicy()
    metric = DownstreamSuccessMetric(env=env, policy=policy, n_episodes=2)
    assert metric.name == "success_rate"
    out = metric.compute()
    assert "success_rate" in out
