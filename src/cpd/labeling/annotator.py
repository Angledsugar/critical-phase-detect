"""Streamlit annotator for RLT-style handoff labels.

Loads a trajectory directory (per-step images or a state-vector .npy + side
matplotlib render), shows a time slider, lets the annotator pick a single
handoff step, and writes the JSON via ``store.save_label``. In pilot mode
(with reference annotations under ``{output_dir}/_reference/``) it computes
Cohen-style kappa after submission so the user knows whether they passed
the calibration bar (k >= 0.8).

Streamlit is imported lazily so other labeling modules can be used without
the labeling extra installed.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cpd.labeling.rubric import RubricV1
from cpd.labeling.store import compute_kappa, load_labels, save_label


def _list_trajectories(traj_dir: Path) -> list[str]:
    """Each subdir of traj_dir is one trajectory (id = subdir name)."""
    if not traj_dir.exists():
        return []
    return sorted(d.name for d in traj_dir.iterdir() if d.is_dir())


def _load_traj_frames(
    traj_subdir: Path,
) -> tuple[Sequence[Any], Sequence[Sequence[float]] | None, str]:
    """Return (frames, actions, mode).

    Three loading paths in order of preference:
      1. ``frames/000.png`` images — returned as PIL/np frames (mode='img')
      2. ``states.npy`` state vectors — fallback render (mode='vec')
      3. Empty — mode='empty'
    Actions returned from ``actions.npy`` if present.
    """
    import numpy as np

    frames_dir = traj_subdir / "frames"
    actions_path = traj_subdir / "actions.npy"
    actions: Sequence[Sequence[float]] | None = None
    if actions_path.exists():
        actions = np.load(actions_path)

    if frames_dir.is_dir():
        import imageio.v3 as iio

        files = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
        frames = [iio.imread(p) for p in files]
        return frames, actions, "img"

    states_path = traj_subdir / "states.npy"
    if states_path.exists():
        states = np.load(states_path)
        return states, actions, "vec"

    return [], actions, "empty"


def _render_state_vector(state: Any, step: int) -> Any:
    """Tiny matplotlib bar plot for a state vector frame fallback."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(state).reshape(-1)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.bar(range(len(arr)), arr)
    ax.set_title(f"step {step} — state vector (d={len(arr)})")
    ax.set_xlabel("dim")
    return fig


def run_annotator(
    traj_dir: Path,
    output_dir: Path,
    annotator_id: str,
    rubric_path: Path | None = None,
    pilot_set: list[str] | None = None,
) -> None:
    """Streamlit entrypoint. Call from ``scripts/run_annotator.py``."""
    import streamlit as st

    traj_dir = Path(traj_dir)
    output_dir = Path(output_dir)

    st.set_page_config(page_title="Handoff Annotator", layout="wide")
    st.title("RLT-style Handoff Annotator")
    st.caption(f"annotator_id = {annotator_id} | rubric v1")

    # Rubric panel.
    with st.expander("Rubric (v1)", expanded=False):
        if rubric_path is not None and Path(rubric_path).exists():
            st.markdown(Path(rubric_path).read_text(encoding="utf-8"))
        else:
            st.markdown(RubricV1().to_markdown())

    # Trajectory selection.
    traj_ids = _list_trajectories(traj_dir)
    if pilot_set is not None:
        pilot_ids = [t for t in traj_ids if t in set(pilot_set)]
        st.info(f"Pilot mode: {len(pilot_ids)} / {len(pilot_set)} pilot trajectories found.")
        traj_ids = pilot_ids
    if not traj_ids:
        st.error(f"no trajectories under {traj_dir}")
        return

    traj_id = st.sidebar.selectbox("Trajectory", traj_ids)
    traj_subdir = traj_dir / traj_id
    frames, actions, mode = _load_traj_frames(traj_subdir)
    if mode == "empty":
        st.warning(f"{traj_subdir} has no frames/ or states.npy — nothing to show.")
        return

    T = len(frames)
    step = st.slider("step", min_value=0, max_value=max(T - 1, 0), value=0)

    col_main, col_side = st.columns([3, 2])
    with col_main:
        st.subheader(f"step {step} / {T - 1}")
        if mode == "img":
            st.image(frames[step], use_column_width=True)
        else:
            st.pyplot(_render_state_vector(frames[step], step))

    with col_side:
        st.subheader("Action vector")
        if actions is not None and step < len(actions):
            st.write(actions[step])
        else:
            st.write("— no actions available —")
        st.subheader("Handoff selection")
        no_handoff = st.checkbox("This trajectory has NO handoff", value=False)
        st.write(f"Selected step: **{-1 if no_handoff else step}**")

        if st.button("Save label", type="primary"):
            handoff_step = -1 if no_handoff else step
            path = save_label(
                output_dir,
                traj_id=traj_id,
                handoff_step=handoff_step,
                annotator_id=annotator_id,
                num_steps=T,
            )
            st.success(f"saved → {path}")

            if pilot_set is not None:
                ref_dir = output_dir / "_reference"
                if ref_dir.is_dir():
                    mine = load_labels(output_dir, annotator_id=annotator_id)
                    ref = load_labels(output_dir, annotator_id="_reference")
                    pilot = set(pilot_set)
                    mine_p = {k: v for k, v in mine.items() if k in pilot}
                    ref_p = {k: v for k, v in ref.items() if k in pilot}
                    if mine_p and ref_p:
                        kappa = compute_kappa(mine_p, ref_p)
                        target = "PASS" if kappa >= 0.8 else "below 0.8"
                        st.info(
                            f"pilot kappa vs _reference: {kappa:.3f} ({target}, "
                            f"n={len(set(mine_p) & set(ref_p))})"
                        )
                    else:
                        st.info("pilot kappa: no overlap with _reference yet.")
                else:
                    st.info("pilot kappa skipped: no _reference/ directory found.")
