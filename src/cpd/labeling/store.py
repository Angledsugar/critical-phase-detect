"""Storage / loader / split for RLT-style handoff annotations.

JSON layout: ``{output_dir}/{annotator_id}/{traj_id}.json`` per label, plus
``{output_dir}/splits.json`` for the frozen 200/100 train/test split. Atomic
writes (write-to-temp + os.replace) so a killed annotator never corrupts a
file.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_label(
    output_dir: Path | str,
    traj_id: str,
    handoff_step: int,
    annotator_id: str,
    **meta: Any,
) -> Path:
    """Atomically write a single label JSON. Returns the file path."""
    out = Path(output_dir) / annotator_id
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{traj_id}.json"

    payload: dict[str, Any] = {
        "traj_id": traj_id,
        "handoff_step": int(handoff_step),
        "annotator_id": annotator_id,
        "rubric_version": meta.pop("rubric_version", "v1"),
        "timestamp": meta.pop("timestamp", _utc_now()),
    }
    payload.update(meta)

    # Atomic write: tmpfile in same dir → os.replace.
    fd, tmp_path = tempfile.mkstemp(prefix=f".{traj_id}.", suffix=".json", dir=str(out))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return target


def load_labels(
    output_dir: Path | str,
    annotator_id: str | None = None,
) -> dict[str, dict]:
    """Load labels.

    If ``annotator_id`` given → ``{traj_id: payload}``.
    Otherwise → ``{annotator_id: {traj_id: payload}}`` (one level deeper).
    Skips ``_reference``-style hidden / underscore-prefixed dirs only when
    they are not explicitly requested.
    """
    root = Path(output_dir)
    if not root.exists():
        return {}

    if annotator_id is not None:
        sub = root / annotator_id
        if not sub.is_dir():
            return {}
        return _load_dir(sub)

    out: dict[str, dict] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        out[child.name] = _load_dir(child)
    return out


def _load_dir(d: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for f in sorted(d.glob("*.json")):
        with open(f, encoding="utf-8") as h:
            payload = json.load(h)
        labels[payload.get("traj_id", f.stem)] = payload
    return labels


def compute_kappa(
    annotations_a: Mapping[str, Mapping[str, Any]],
    annotations_b: Mapping[str, Mapping[str, Any]],
    tolerance: int = 3,
) -> float:
    """Cohen-style kappa on per-trajectory handoff agreement.

    Each trajectory yields a binary outcome:
        agree iff ``|step_a - step_b| <= tolerance``  (or both -1 for "no
        handoff").
    We then build a 2x2 table over the shared traj IDs (agree / disagree
    along both raters' "internally consistent vs not" — modeled as each
    rater voting agree-with-self; empty intersections short-circuit).

    Practical interpretation: with binary "is this trajectory's handoff
    locked in?" outcome, kappa reduces to the standard Cohen formula
    treating ``a`` and ``b`` as raters of "do you agree with the consensus
    step (within tolerance)?". This gives 1.0 when identical, ~0 when
    independent, and lies in [-1, 1] otherwise.
    """
    shared = sorted(set(annotations_a) & set(annotations_b))
    if not shared:
        return 0.0

    # Per-traj agreement vector and a synthetic per-rater "claim"
    # (rater A's claim is fixed True; rater B's claim is True iff within
    # tolerance). This mirrors RLT's annotator-vs-reference setup.
    agree = 0
    a_pos = len(shared)  # rater A always claims a handoff (the one they marked)
    b_pos = 0
    for tid in shared:
        sa = int(annotations_a[tid]["handoff_step"])
        sb = int(annotations_b[tid]["handoff_step"])
        if sa == -1 and sb == -1:
            agree += 1
            b_pos += 1  # both agree there is no handoff -> "agree" claim
            a_pos -= 1
            continue
        if sa == -1 or sb == -1:
            continue
        if abs(sa - sb) <= tolerance:
            agree += 1
            b_pos += 1

    n = len(shared)
    p_o = agree / n
    # Marginals for chance agreement: rater A's positive rate and rater
    # B's positive rate (positive == "handoff present and within tol").
    pa = a_pos / n
    pb = b_pos / n
    p_e = pa * pb + (1 - pa) * (1 - pb)
    if p_e >= 1.0:  # degenerate — perfect agreement on a constant signal
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def split_dataset(
    label_paths: Iterable[str] | Mapping[str, Any],
    train: int = 200,
    test: int = 100,
    seed: int = 0,
    output_dir: Path | str | None = None,
) -> dict[str, list[str]]:
    """Build a frozen train/test split over annotated traj IDs.

    Accepts an iterable of traj IDs, an iterable of JSON paths, or a
    ``{traj_id: payload}`` mapping (output of ``load_labels`` for one
    annotator). When ``output_dir`` is given, writes ``splits.json`` there.
    Raises ``ValueError`` when fewer than ``train + test`` IDs are
    available.
    """
    ids = _coerce_ids(label_paths)
    if len(ids) < train + test:
        raise ValueError(
            f"need at least {train + test} labeled trajectories, got {len(ids)}"
        )

    rng = random.Random(seed)
    shuffled = sorted(ids)  # determinism in face of dict order
    rng.shuffle(shuffled)
    train_ids = sorted(shuffled[:train])
    test_ids = sorted(shuffled[train : train + test])

    splits = {"train": train_ids, "test": test_ids, "seed": seed}

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "splits.json", "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2, sort_keys=True)

    return splits


def _coerce_ids(x: Iterable[str] | Mapping[str, Any]) -> list[str]:
    if isinstance(x, Mapping):
        return list(x.keys())
    out: list[str] = []
    for item in x:
        p = Path(str(item))
        # Heuristic: if it looks like a path with a JSON suffix, use stem.
        out.append(p.stem if p.suffix == ".json" else str(item))
    return out


def load_split(
    splits_path: Path | str,
    split: Literal["train", "test"],
    labels_root: Path | str | None = None,
    annotator_id: str | None = None,
) -> dict[str, int]:
    """Return ``{traj_id: handoff_step}`` for the requested split.

    Reads ``splits.json``, then materialises handoff steps from
    ``labels_root`` (defaults to the splits file's parent) under
    ``annotator_id`` if given, else the first annotator dir found.
    """
    splits_path = Path(splits_path)
    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)
    if split not in splits:
        raise KeyError(f"split '{split}' not in {splits_path}: keys={list(splits)}")
    traj_ids: list[str] = list(splits[split])

    root = Path(labels_root) if labels_root is not None else splits_path.parent
    if annotator_id is None:
        candidates = sorted(
            d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_")
        )
        if not candidates:
            raise FileNotFoundError(f"no annotator subdir under {root}")
        annotator_id = candidates[0].name

    labels = load_labels(root, annotator_id=annotator_id)
    out: dict[str, int] = {}
    for tid in traj_ids:
        if tid not in labels:
            raise KeyError(f"trajectory {tid!r} in split but not in labels")
        out[tid] = int(labels[tid]["handoff_step"])
    return out
