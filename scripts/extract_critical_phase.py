"""Critical-phase extraction per paper plan §3.5 / §5.2.

CPD score = r_t = log f̃+(φ(s_t)) − log f̃−(φ(s_t))   (paper plan §5.2 r^(A))
Critical phase = contiguous intervals {t : r_t < τ}   (τ default = 0; paper's
log-ratio sign — negative means "more likely from failure buffer").

Outputs (under reports/exp1/critical_phase/):
  - per_episode.json     : intervals, longest run, t_min (argmin), summary per ep
  - fig4_<ep>.png        : reward curve with critical intervals shaded + key frames strip
  - <ep>_cp.mp4          : video with red border overlay during critical phase
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import imageio.v2 as imageio
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


def encode(encoder: TLDREncoder, raw: np.ndarray, scale: float) -> torch.Tensor:
    with torch.no_grad():
        return encoder.encode(raw).detach() / scale


def find_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) intervals where mask is True."""
    if not mask.any():
        return []
    diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def annotate_frame(
    agent: np.ndarray,
    wrist: np.ndarray,
    *,
    step: int,
    total: int,
    success: bool,
    label: str,
    is_critical: bool,
    r_t: float,
) -> np.ndarray:
    side = np.concatenate([agent, wrist], axis=1)
    h, w = side.shape[:2]
    if is_critical:
        cv2.rectangle(side, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
    cv2.putText(side, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(side, f"step {step:03d}/{total:03d}", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    color = (0, 200, 0) if success else (0, 60, 220)
    tag = "SUCCESS" if success else "FAILURE"
    cv2.putText(side, tag, (w - 130, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(side, f"r_t={r_t:+6.2f}", (w - 130, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if is_critical:
        cv2.putText(side, "CRITICAL", (w // 2 - 60, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return side


def render_cp_mp4(
    npz_path: Path, out_path: Path, *, rewards: np.ndarray, cp_mask: np.ndarray, fps: int
) -> None:
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
                agent[t], wrist[t], step=t, total=T, success=success, label=label,
                is_critical=bool(cp_mask[t]), r_t=float(rewards[t]),
            )
            writer.append_data(frame)
    finally:
        writer.close()


def plot_fig4(
    *, npz_path: Path, rewards: np.ndarray, cp_mask: np.ndarray, intervals, out_png: Path,
    success: bool, label: str,
) -> None:
    npz = np.load(npz_path)
    agent = npz["agentview"]
    T = rewards.shape[0]
    t_axis = np.arange(T)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 5, height_ratios=[2, 1], hspace=0.3, wspace=0.15)
    ax = fig.add_subplot(gs[0, :])
    color = "tab:green" if success else "tab:red"
    ax.plot(t_axis, rewards, color=color, lw=1.2, label=f"r_t  ({'SUCCESS' if success else 'FAILURE'})")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    for s, e in intervals:
        ax.axvspan(s, e - 1, color="tab:red", alpha=0.18,
                   label="critical phase (r_t<0)" if (s, e) == intervals[0] else None)
    if intervals:
        argmin_t = int(np.argmin(rewards))
        ax.axvline(argmin_t, color="tab:red", lw=1.0, ls="--",
                   label=f"argmin r_t = {argmin_t}")
    ax.set_xlabel("step t")
    ax.set_ylabel("r_t = log f+(z_t) − log f−(z_t)")
    ax.set_title(f"{npz_path.stem} — {label}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # key-frame strip: 5 frames at quantile positions of cp_mask (or even spread if no cp)
    if cp_mask.any():
        cp_idx = np.where(cp_mask)[0]
        picks = np.linspace(0, len(cp_idx) - 1, 5).astype(int)
        sample_t = cp_idx[picks]
    else:
        sample_t = np.linspace(0, T - 1, 5).astype(int)
    for i, t in enumerate(sample_t):
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(agent[t])
        ax2.set_xticks([]); ax2.set_yticks([])
        is_crit = bool(cp_mask[t])
        edge = "red" if is_crit else "gray"
        for spine in ax2.spines.values():
            spine.set_edgecolor(edge); spine.set_linewidth(2)
        ax2.set_title(f"t={int(t)}  r={rewards[int(t)]:+.1f}", fontsize=9,
                      color="red" if is_crit else "black")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default="data/tldr_demos.pkl")
    p.add_argument("--encoder", default="checkpoints/tldr.pt")
    p.add_argument("--rollouts",
                   default="/media/engineer/DATA/datasets/cpd_rollouts/pi05_libero/libero_10/task00")
    p.add_argument("--out", default="reports/exp1/critical_phase")
    p.add_argument("--tau", type=float, default=0.0,
                   help="critical-phase threshold: r_t < tau ⇒ critical. Default 0 (log-ratio sign).")
    p.add_argument("--min-run", type=int, default=3,
                   help="drop intervals shorter than this many steps (debounce).")
    p.add_argument("--render-mp4", action="store_true", default=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episodes", type=str, default=None,
                   help="comma-separated episode stems (e.g. ep115,ep118) to restrict "
                        "rendering to. KDE buffer still uses ALL rollouts in --rollouts "
                        "(LOO against the full set). Default: all episodes.")
    args = p.parse_args()
    only = set(args.episodes.split(",")) if args.episodes else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # encoder + scale
    encoder = TLDREncoder.load(args.encoder)
    encoder.eval()
    with Path(args.demos).open("rb") as f:
        demo_raw: list[np.ndarray] = pickle.load(f)
    raw_finals = torch.stack([encoder.encode(r)[-1].detach() for r in demo_raw])
    scale = float(raw_finals.norm(dim=-1).mean())

    # build buffer with GT-labeled rollouts (paper plan §3.5 needs both buckets)
    rollouts = []
    for fp in sorted(Path(args.rollouts).glob("ep*.npz")):
        npz = np.load(fp)
        rollouts.append({
            "path": fp, "state": npz["state"],
            "success": bool(npz["success"]),
            "length": int(npz["length"]),
        })
    if not rollouts:
        raise SystemExit(f"no rollouts under {args.rollouts}")

    rollout_trajs = [
        Trajectory(
            raw_states=[r.tolist() for r in r0["state"]],
            latents=encode(encoder, r0["state"], scale),
            success=r0["success"],
        )
        for r0 in rollouts
    ]
    # Paper plan §10 risk: with N=10 rollouts and N-=2 failures, KDE's B- is
    # dominated by self-density when evaluating ep5 / ep9 on itself. Leave-one-out:
    # for each rollout, rebuild KDE from the *other* 9 — i.e. exclude the
    # trajectory we're scoring. This is what Theorem 1 implicitly assumes
    # (rollout under eval is out-of-sample w.r.t. the buffer).
    print(f"[cp] LOO mode: building one KDE per rollout from the other {len(rollout_trajs)-1} trajs")

    summary = []
    for idx, (r0, traj) in enumerate(zip(rollouts, rollout_trajs)):
        if only is not None and r0["path"].stem not in only:
            continue
        buf_loo = TrajectoryBuffer()
        for j, other in enumerate(rollout_trajs):
            if j == idx:
                continue
            buf_loo.add(other)
        if buf_loo.n_positive == 0 or buf_loo.n_negative == 0:
            print(f"  [warn] {r0['path'].stem}: LOO buffer degenerate "
                  f"(B+={buf_loo.n_positive}, B-={buf_loo.n_negative}); skipping")
            continue
        kde_loo = compute_kde(buf_loo)
        reward_loo = Reward(kde=kde_loo)
        rs = reward_loo.per_step(traj.latents).cpu().numpy()
        mask = rs < args.tau
        intervals = find_intervals(mask)
        intervals = [(s, e) for s, e in intervals if (e - s) >= args.min_run]
        # rebuild mask after min-run filter
        clean_mask = np.zeros_like(mask)
        for s, e in intervals:
            clean_mask[s:e] = True
        longest = max((e - s for s, e in intervals), default=0)
        first_t = intervals[0][0] if intervals else None
        argmin_t = int(np.argmin(rs))

        ep_id = r0["path"].stem
        plot_fig4(
            npz_path=r0["path"], rewards=rs, cp_mask=clean_mask, intervals=intervals,
            out_png=out_dir / f"fig4_{ep_id}.png",
            success=r0["success"],
            label=str(np.load(r0["path"])["task_description"]),
        )
        if args.render_mp4:
            render_cp_mp4(
                r0["path"], out_dir / f"{ep_id}_cp.mp4",
                rewards=rs, cp_mask=clean_mask, fps=args.fps,
            )

        info = {
            "episode": ep_id,
            "success": r0["success"],
            "T": r0["length"],
            "n_critical_steps": int(clean_mask.sum()),
            "critical_fraction": float(clean_mask.mean()),
            "first_critical_t": first_t,
            "argmin_r_t": argmin_t,
            "min_r_t": float(rs.min()),
            "mean_r_t": float(rs.mean()),
            "sum_r_t": float(rs.sum()),
            "longest_critical_run": int(longest),
            "intervals": [[int(s), int(e)] for s, e in intervals],
            "kde": {
                "h": float(kde_loo.h),
                "n_pos": int(kde_loo.latents_pos.shape[0]),
                "n_neg": int(kde_loo.latents_neg.shape[0]),
            },
        }
        summary.append(info)
        print(
            f"  {ep_id}  T={info['T']:3d}  succ={info['success']!s:5}  "
            f"crit_steps={info['n_critical_steps']:3d} ({info['critical_fraction']:.2%})  "
            f"longest={info['longest_critical_run']:3d}  "
            f"argmin t={info['argmin_r_t']:3d}  min r={info['min_r_t']:+7.2f}"
        )

    (out_dir / "per_episode.json").write_text(json.dumps(summary, indent=2))
    print(f"[cp] saved → {out_dir/'per_episode.json'}")


if __name__ == "__main__":
    main()
