"""1st experiment result: TLDR + G2 + KDE + log(f+/f-) reward on LIBERO-Long task 0.

Inputs:
  - data/tldr_demos.pkl     : 500 LIBERO-Long expert demos (proprio (T,8))
  - checkpoints/tldr.pt     : TLDR encoder trained on the above
  - /media/engineer/DATA/datasets/cpd_rollouts/pi05_libero/libero_10/task00/
                            : 10 pi05 rollouts (npz w/ state, success, ...)

Outputs (saved under reports/exp1/):
  - metrics.json            : G2 vs ground-truth confusion matrix, reward separability
  - phi_pca.png             : 2-D PCA of φ(s_t) for demo finals + rollout finals
  - reward_curves.png       : per-step log(f+/f-) over time, success vs failure
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.kde import compute_kde
from cpd.core.labeler import G2Labeler
from cpd.core.reward import Reward
from cpd.core.trajectory import Trajectory
from cpd.encoders.tldr import TLDREncoder


def _encode_traj(encoder: TLDREncoder, raw: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        return encoder.encode(raw).detach()


def _make_traj(
    encoder: TLDREncoder, raw: np.ndarray, *, success: bool | None, scale: float = 1.0
) -> Trajectory:
    z = _encode_traj(encoder, raw) / max(scale, 1e-9)
    return Trajectory(
        raw_states=[r.tolist() for r in raw],
        latents=z,
        success=success,
        metadata={"length": int(raw.shape[0])},
    )


def load_rollouts(rollout_dir: Path) -> list[dict]:
    out = []
    for fp in sorted(rollout_dir.glob("ep*.npz")):
        npz = np.load(fp)
        out.append(
            {
                "path": fp,
                "state": npz["state"],          # (T, 9 typically) — pi05 client proprio
                "success": bool(npz["success"]),
                "length": int(npz["length"]),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default="data/tldr_demos.pkl")
    p.add_argument("--encoder", default="checkpoints/tldr.pt")
    p.add_argument(
        "--rollouts",
        default="/media/engineer/DATA/datasets/cpd_rollouts/pi05_libero/libero_10/task00",
    )
    p.add_argument("--out", default="reports/exp1")
    p.add_argument("--quantile", type=float, default=0.95)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Encoder
    encoder = TLDREncoder.load(args.encoder)
    encoder.eval()
    print(f"[exp1] encoder: state_dim={encoder.state_dim} latent_dim={encoder.latent_dim}")

    # 2. Demo trajectories (proprio dim 8). First pass to compute global latent
    # scale (mean ‖φ(s_T)‖ across demos) so that downstream KDE doesn't underflow
    # in the 64-dim latent space.
    with Path(args.demos).open("rb") as f:
        demo_raw: list[np.ndarray] = pickle.load(f)
    raw_finals = []
    for r in demo_raw:
        z = _encode_traj(encoder, r)
        raw_finals.append(z[-1])
    raw_final_norm = float(torch.stack(raw_finals).norm(dim=-1).mean())
    print(f"[exp1] mean ‖φ(s_T)‖ on demos = {raw_final_norm:.3f} → using as latent scale")

    demo_trajs = [_make_traj(encoder, r, success=True, scale=raw_final_norm) for r in demo_raw]
    print(f"[exp1] demos: {len(demo_trajs)} trajectories (raw_dim={demo_raw[0].shape[1]})")

    # 3. Rollout trajectories. pi05 client proprio = ee_pos(3)+axisangle(3)+gripper(2) = 8.
    rollouts = load_rollouts(Path(args.rollouts))
    if not rollouts:
        raise SystemExit(f"no rollouts under {args.rollouts}")
    rollout_dim = rollouts[0]["state"].shape[1]
    print(f"[exp1] rollouts: {len(rollouts)} (raw_dim={rollout_dim})")
    assert rollout_dim == encoder.state_dim, (
        f"rollout dim {rollout_dim} != encoder dim {encoder.state_dim}; "
        "demo proprio (ee_pos+ee_ori+gripper_states) and rollout proprio "
        "(eef_pos+axisangle(quat)+gripper_qpos) must align."
    )
    rollout_trajs = []
    rollout_gt = []
    for r in rollouts:
        rollout_trajs.append(_make_traj(encoder, r["state"], success=None, scale=raw_final_norm))
        rollout_gt.append(r["success"])

    # 4. G2 self-labeler from demos
    labeler = G2Labeler.from_demos(demo_trajs, quantile=args.quantile)
    print(f"[exp1] G2: ‖g‖={float(labeler.g.norm()):.3f}  ε(quantile={args.quantile})={labeler.epsilon:.3f}")

    # 5. Apply G2 to rollouts → predicted success
    preds = []
    g2_dists = []
    for traj in rollout_trajs:
        labeled = labeler.label(traj)
        preds.append(bool(labeled.success))
        g2_dists.append(float(torch.linalg.norm(traj.final_latent - labeler.g)))
    preds = np.asarray(preds)
    gt = np.asarray(rollout_gt)
    tp = int(((preds) & (gt)).sum())
    fp = int(((preds) & (~gt)).sum())
    tn = int((~preds & ~gt).sum())
    fn = int((~preds & gt).sum())
    n = len(gt)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / n
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    print(
        f"[exp1] G2 labels vs GT (env success): "
        f"acc={accuracy:.2f} prec={precision:.2f} recall={recall:.2f} F1={f1:.3f}  "
        f"(TP={tp} FP={fp} TN={tn} FN={fn})"
    )

    # 6. Build buffer using G2 self-labels (paper §3.1: B+ from G2-positive trajs)
    buf_g2 = TrajectoryBuffer()
    for traj in rollout_trajs:
        buf_g2.add(labeler.label(traj))
    print(f"[exp1] G2 buffer: B+={buf_g2.n_positive} B-={buf_g2.n_negative}")

    # GT-labeled buffer used for the reward demonstration (1st-result definition):
    # paper §3.5 reward = log f+ − log f− needs both buckets non-empty. With only
    # 10 rollouts and high pi05 success rate, G2 rarely produces B-. Use env
    # success for the buffer here; G2 vs GT comparison stays as a separate metric.
    buf = TrajectoryBuffer()
    for traj, success in zip(rollout_trajs, gt):
        buf.add(traj.with_label(success=bool(success)))
    print(f"[exp1] GT buffer: B+={buf.n_positive} B-={buf.n_negative} (used for KDE)")

    kde = compute_kde(buf)
    reward = Reward(kde=kde)
    print(f"[exp1] KDE bandwidth h={kde.h:.4f}  |B+ latents|={kde.latents_pos.shape[0]} "
          f"|B- latents|={kde.latents_neg.shape[0]}")

    # 8. Per-step rewards per rollout
    per_step = []
    traj_total = []
    for traj in rollout_trajs:
        rs = reward.per_step(traj.latents).cpu().numpy()
        per_step.append(rs)
        traj_total.append(float(rs.sum()))
    per_step = np.asarray(per_step, dtype=object)
    succ_mask = gt
    succ_R = np.array([traj_total[i] for i in range(n) if succ_mask[i]])
    fail_R = np.array([traj_total[i] for i in range(n) if not succ_mask[i]])
    sep = (succ_R.mean() - fail_R.mean()) / (succ_R.std() + fail_R.std() + 1e-9)
    print(
        f"[exp1] sum reward: succ={succ_R.mean():.2f}±{succ_R.std():.2f} "
        f"fail={fail_R.mean():.2f}±{fail_R.std():.2f}  "
        f"separability(z)={sep:.2f}"
    )

    # 9. PCA visualization of φ(s_t)
    demo_finals = torch.stack([t.final_latent for t in demo_trajs]).cpu().numpy()
    rollout_finals = np.stack([t.final_latent.cpu().numpy() for t in rollout_trajs])
    all_finals = np.concatenate([demo_finals, rollout_finals], axis=0)
    mean = all_finals.mean(axis=0, keepdims=True)
    centered = all_finals - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pcs = centered @ Vt[:2].T  # (N, 2)

    fig, ax = plt.subplots(figsize=(7, 6))
    n_demo = demo_finals.shape[0]
    ax.scatter(pcs[:n_demo, 0], pcs[:n_demo, 1], s=10, alpha=0.4, c="tab:gray", label=f"demos (N={n_demo})")
    succ_idx = [n_demo + i for i in range(n) if succ_mask[i]]
    fail_idx = [n_demo + i for i in range(n) if not succ_mask[i]]
    ax.scatter(pcs[succ_idx, 0], pcs[succ_idx, 1], s=80, c="tab:green", marker="o",
               edgecolors="black", linewidths=0.5, label=f"rollout success (N={len(succ_idx)})")
    ax.scatter(pcs[fail_idx, 0], pcs[fail_idx, 1], s=80, c="tab:red", marker="X",
               edgecolors="black", linewidths=0.5, label=f"rollout failure (N={len(fail_idx)})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("φ(s_T) — terminal latents (G2 goal space)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "phi_pca.png", dpi=130)
    plt.close(fig)

    # 10. Per-step reward curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n):
        rs = per_step[i]
        t = np.linspace(0, 1, len(rs))
        c = "tab:green" if succ_mask[i] else "tab:red"
        a = 0.55 if succ_mask[i] else 0.85
        lw = 1.4 if not succ_mask[i] else 1.0
        ax.plot(t, rs, c=c, alpha=a, lw=lw)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="tab:green", label=f"success (N={int(succ_mask.sum())})"),
        Line2D([0], [0], color="tab:red",   label=f"failure (N={int((~succ_mask).sum())})"),
    ], loc="best")
    ax.set_xlabel("normalized time t/T")
    ax.set_ylabel("log f+(z_t) − log f−(z_t)")
    ax.set_title("Per-step CPD reward (KDE log-ratio)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "reward_curves.png", dpi=130)
    plt.close(fig)

    # 11. Save metrics
    metrics = {
        "n_demos": int(len(demo_trajs)),
        "n_rollouts": int(n),
        "encoder": {
            "state_dim": int(encoder.state_dim),
            "latent_dim": int(encoder.latent_dim),
        },
        "g2": {
            "g_norm": float(labeler.g.norm()),
            "epsilon": float(labeler.epsilon),
            "quantile": float(args.quantile),
            "rollout_g2_distance": [float(d) for d in g2_dists],
        },
        "labels": {
            "ground_truth": [bool(x) for x in gt],
            "g2_predicted": [bool(x) for x in preds],
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "kde": {
            "bandwidth_h": float(kde.h),
            "n_pos_latents": int(kde.latents_pos.shape[0]),
            "n_neg_latents": int(kde.latents_neg.shape[0]) if kde.latents_neg.ndim > 0 else 0,
        },
        "reward": {
            "trajectory_total": [float(r) for r in traj_total],
            "success_mean": float(succ_R.mean()) if succ_R.size else None,
            "success_std":  float(succ_R.std())  if succ_R.size else None,
            "fail_mean":    float(fail_R.mean()) if fail_R.size else None,
            "fail_std":     float(fail_R.std())  if fail_R.size else None,
            "separability_z": float(sep),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[exp1] saved metrics → {out_dir/'metrics.json'}")
    print(f"[exp1] saved figures → {out_dir/'phi_pca.png'}, {out_dir/'reward_curves.png'}")


if __name__ == "__main__":
    main()
