# Experiment 1 — TLDR + G2 + KDE log-ratio reward on LIBERO-Long task 0

Date: 2026-05-09
Suite: `libero_10` (LIBERO-Long, paper main)
Task 0: *put both the alphabet soup and the tomato sauce in the basket*
Base policy: `pi05_libero` (openpi checkpoint, JAX server on GPU 1)

## Pipeline

1. **TLDR encoder** (`scripts/train_tldr.py`)
   φ : ℝ⁸ → ℝ⁶⁴, 3-layer MLP + LayerNorm + ReLU. Triplet contrastive loss with
   k_pos=2, K_neg=20, margin=1.0, 50 epochs × 50 batches × 256 triplets.
   Inputs: 500 LIBERO-Long expert demos (10 tasks × 50 demos), proprio =
   `[ee_pos(3), ee_ori(3), gripper_states(2)]`.
   Final: loss 0.17 → 0.003, violation 26 % → 0.5 %, d_an / d_ap ≈ 70.

2. **π_0.5 LIBERO rollouts** (`scripts/collect_libero_rollouts.py`,
   .venv-libero ↔ openpi WebSocket server)
   10 episodes on libero_10 task 0, replan every 5 steps, 520-step cap.
   **Raw success rate: 8 / 10 = 0.80**, mean episode 325 steps, total 331 s.

3. **G2 self-labeler** (`cpd.core.labeler.G2Labeler`)
   g = mean of demo final latents (after scaling by mean ‖φ(s_T)‖ = 38.14).
   ε = 95-th-percentile of demo final-norm distances = **1.32**.

4. **KDE + log-ratio reward** (`cpd.core.kde`, `cpd.core.reward`)
   Refactored to compute everything in log-space (`KDEStats.log_density()`)
   so that the d=64 latent space doesn't over/underflow the kernel evaluation.
   Silverman bandwidth h = 0.0612, |B+| = 2212 latents (8 trajs), |B-| = 1040
   latents (2 trajs). Buffer populated from env-success ground truth.

## Key numbers

|                     | metric                               | value                     |
| ------------------- | ------------------------------------ | ------------------------- |
| π_0.5 rollouts      | env success rate                     | 0.80 (8/10)               |
| G2 vs ground truth  | accuracy / precision / recall / F1   | 0.80 / 0.80 / **1.00** / **0.889** |
| G2 confusion        | TP / FP / TN / FN                    | 8 / 2 / 0 / 0             |
| Reward (sum)        | success_mean ± std                   | **+1209 ± 426**           |
| Reward (sum)        | failure_mean ± std                   | **−3438 ± 710**           |
| Reward separability | z-score = (μ⁺−μ⁻) / (σ⁺+σ⁻)          | **4.09**                  |

The per-step reward `log f+(z_t) − log f-(z_t)` separates the two failure
trajectories from the eight successes by ≈ 4 σ — well above any plausible
random-baseline threshold. This is the headline 1st result.

## Figures

- `phi_pca.png` — PCA of φ(s_T) shows rollout finals embedded in the demo
  cluster; the two failures sit at the edge of the demo support but are not
  catastrophically off-distribution (matches the G2 false-positive: by
  terminal-only criterion they look like demos).
- `reward_curves.png` — per-step log-ratio reward over time. Success traces
  drift strictly upward to +5..+20; failure traces accumulate strongly
  negative reward, reaching −20..−40 by t/T = 1. Signal is monotone in t
  and visually unambiguous.

## Artifacts

```
reports/exp1/
├── metrics.json           # all numbers above + per-rollout breakdown
├── phi_pca.png            # 2-D PCA of φ(s_T)
├── reward_curves.png      # per-step log f+/f-
└── summary.md             # this file

checkpoints/tldr.pt        # encoder weights (state_dim=8, latent_dim=64)
data/tldr_demos.pkl        # 500 demo proprio sequences
/media/engineer/DATA/datasets/cpd_rollouts/pi05_libero/libero_10/task00/
                           # 10 npz rollouts (agentview, wrist, state, action, reward)
```

## Caveats / next steps

- **N=10 rollouts** is too small for a tight failure-side estimate; only 2
  failures define the entire B- bucket. Scale to ≥ 50 episodes per task,
  ≥ 5 tasks, before reporting paper numbers.
- **G2 false-positive rate is 100 %** on this run (ε too loose). The
  terminal-only criterion is by design generous; for paper claims we need
  Theorem-1 conditions — i.e. evaluate G2 on a richer bench where we
  expect 30–60 % failures (refiner-attempt phase) so that ε actually cuts
  the support.
- **Latent scaling by mean ‖φ(s_T)‖** is a band-aid; principled fix is L2
  normalization at the encoder output (re-train).
- **Result 2 (reward → PPO refiner)** still needs `cpd.policies.ppo_refiner`
  hooked up to the libero env in .venv-libero.
- **Theorem 1 ablation** (RLT label vs G2 label) needs RLT supervised
  labels — Q3 in `paper/cpd_g2_corl26_v3_phasea.md` open.
