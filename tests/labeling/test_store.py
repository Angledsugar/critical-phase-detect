"""save_label / load_labels / compute_kappa / split_dataset roundtrip.

GUI annotator behaviour is out of scope here — only data ops.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from cpd.labeling.store import (
    compute_kappa,
    load_labels,
    load_split,
    save_label,
    split_dataset,
)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    save_label(tmp_path, traj_id="traj_001", handoff_step=42, annotator_id="alice")
    save_label(tmp_path, traj_id="traj_002", handoff_step=-1, annotator_id="alice")

    loaded = load_labels(tmp_path, annotator_id="alice")
    assert set(loaded) == {"traj_001", "traj_002"}
    assert loaded["traj_001"]["handoff_step"] == 42
    assert loaded["traj_002"]["handoff_step"] == -1
    assert loaded["traj_001"]["annotator_id"] == "alice"
    assert loaded["traj_001"]["rubric_version"] == "v1"
    assert "timestamp" in loaded["traj_001"]


def test_save_label_passes_meta(tmp_path: Path) -> None:
    save_label(
        tmp_path,
        traj_id="t",
        handoff_step=5,
        annotator_id="alice",
        num_steps=120,
        notes="contact ambiguity",
    )
    loaded = load_labels(tmp_path, annotator_id="alice")["t"]
    assert loaded["num_steps"] == 120
    assert loaded["notes"] == "contact ambiguity"


def test_save_label_atomic(tmp_path: Path) -> None:
    """No leftover .tmp file in the annotator dir."""
    save_label(tmp_path, traj_id="t", handoff_step=3, annotator_id="alice")
    sub = tmp_path / "alice"
    leftovers = [p.name for p in sub.iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_load_labels_all_annotators(tmp_path: Path) -> None:
    save_label(tmp_path, traj_id="t1", handoff_step=1, annotator_id="alice")
    save_label(tmp_path, traj_id="t1", handoff_step=2, annotator_id="bob")
    all_labels = load_labels(tmp_path)
    assert set(all_labels) == {"alice", "bob"}
    assert all_labels["alice"]["t1"]["handoff_step"] == 1
    assert all_labels["bob"]["t1"]["handoff_step"] == 2


def test_load_labels_missing_dir(tmp_path: Path) -> None:
    assert load_labels(tmp_path / "nope") == {}
    assert load_labels(tmp_path, annotator_id="ghost") == {}


def _ann(steps: dict[str, int]) -> dict[str, dict]:
    return {tid: {"traj_id": tid, "handoff_step": s} for tid, s in steps.items()}


def test_kappa_identical_is_one() -> None:
    a = _ann({"t1": 10, "t2": 20, "t3": 30, "t4": 40})
    assert compute_kappa(a, a, tolerance=0) == pytest.approx(1.0)
    assert compute_kappa(a, a, tolerance=3) == pytest.approx(1.0)


def test_kappa_random_near_zero() -> None:
    rng = random.Random(0)
    n = 400
    a = _ann({f"t{i}": rng.randint(0, 200) for i in range(n)})
    b = _ann({f"t{i}": rng.randint(0, 200) for i in range(n)})
    k = compute_kappa(a, b, tolerance=3)
    # Two independent uniform-on-[0,200] integers within ±3 -> ~7/200 chance.
    # Both rater "claim" rates collapse near that, kappa hovers near 0.
    assert -0.15 < k < 0.15


def test_kappa_partial_in_unit_interval() -> None:
    a = _ann({"t1": 10, "t2": 20, "t3": 30, "t4": 40})
    b = _ann({"t1": 10, "t2": 21, "t3": 50, "t4": 41})  # 3 of 4 within tol=3
    k = compute_kappa(a, b, tolerance=3)
    assert 0.0 <= k <= 1.0


def test_kappa_no_overlap_returns_zero() -> None:
    a = _ann({"t1": 10})
    b = _ann({"t2": 10})
    assert compute_kappa(a, b) == 0.0


def test_kappa_no_handoff_both_agree() -> None:
    a = _ann({"t1": -1, "t2": -1})
    b = _ann({"t1": -1, "t2": -1})
    assert compute_kappa(a, b) == pytest.approx(1.0)


def test_split_dataset_disjoint_and_sized(tmp_path: Path) -> None:
    ids = [f"t{i:04d}" for i in range(300)]
    splits = split_dataset(ids, train=200, test=100, seed=0, output_dir=tmp_path)
    assert len(splits["train"]) == 200
    assert len(splits["test"]) == 100
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert (tmp_path / "splits.json").exists()
    on_disk = json.loads((tmp_path / "splits.json").read_text())
    assert on_disk["train"] == splits["train"]
    assert on_disk["test"] == splits["test"]


def test_split_dataset_deterministic() -> None:
    ids = [f"t{i:04d}" for i in range(300)]
    s1 = split_dataset(ids, train=200, test=100, seed=42)
    s2 = split_dataset(ids, train=200, test=100, seed=42)
    s3 = split_dataset(ids, train=200, test=100, seed=43)
    assert s1["train"] == s2["train"]
    assert s1["train"] != s3["train"]


def test_split_dataset_too_few() -> None:
    with pytest.raises(ValueError, match="at least 300"):
        split_dataset([f"t{i}" for i in range(50)], train=200, test=100)


def test_split_dataset_accepts_mapping(tmp_path: Path) -> None:
    mapping = {f"t{i:04d}": {"handoff_step": i} for i in range(300)}
    splits = split_dataset(mapping, train=200, test=100, seed=0)
    assert len(splits["train"]) == 200
    assert len(splits["test"]) == 100


def test_load_split_returns_handoffs(tmp_path: Path) -> None:
    for i in range(300):
        save_label(
            tmp_path,
            traj_id=f"t{i:04d}",
            handoff_step=i,
            annotator_id="alice",
        )
    labels = load_labels(tmp_path, annotator_id="alice")
    split_dataset(labels, train=200, test=100, seed=0, output_dir=tmp_path)

    train = load_split(tmp_path / "splits.json", "train", labels_root=tmp_path)
    test = load_split(tmp_path / "splits.json", "test", labels_root=tmp_path)
    assert len(train) == 200 and len(test) == 100
    assert set(train).isdisjoint(test)
    sample_id, sample_step = next(iter(train.items()))
    assert sample_step == int(sample_id[1:])  # "t0042" -> 42
