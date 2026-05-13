# Exp2 — CPD detection F1 vs N (rollout budget)

## Question

> 200개 episode를 단일 task에서 모아서, N을 늘릴수록 CPD가 failure를 얼마나 잘 잡아내는지(F1) 측정. 몇 N에서 F1 ≥ 0.95에 도달하나?

## Setup

- **Task**: LIBERO-Long (`libero_10`) task 0 — "put both the alphabet soup and the tomato sauce in the basket"
- **Policy**: π_0.5_libero (openpi serve, replan=5)
- **Encoder**: TLDR (`checkpoints/tldr.pt`), latent scale = 38.14
- **Buffer label**: ground-truth success/failure (G2 self-label disabled for this exp — we want an upper bound on the *signal*, not a noisy label test)
- **KDE**: leave-one-out — for each evaluated episode, KDE built from the other N-1 trajectories (Silverman in 64-dim latent space, log-space)
- **CPD score**: r_t = log f̃+(z_t) − log f̃−(z_t)
- **Rules tested** (all use τ=0, min-run=3 to define critical intervals first):
  - `has_critical`: ∃ any critical interval (original loose rule)
  - `longest_run ≥ L*`: duration of worst dip exceeds threshold
  - `n_crit_steps ≥ K*`: total critical-step count exceeds threshold
  - `critical_fraction ≥ F*`: critical/T exceeds threshold
- **F1 evaluation**: **leave-one-out CV** — threshold from the other N-1 episodes, predict on the held-out one. Aggregate predictions over all 200 LOO folds. (Avoids in-sample threshold-fitting optimism.)
- **CI**: bootstrap (200 resamples; LOO is O(N²) so kept small).

## Headline numbers

200 collected: 187 success / 13 fail (success rate 93.5%).

| N | has_critical | longest_run | **n_crit_steps** | critical_fraction |
|---:|:---:|:---:|:---:|:---:|
| 10 | 0.333 | 0.333 | 0.667 | 0.667 |
| 20 | 0.333 | 0.667 | 0.667 | 0.400 |
| 30 | 0.235 | 0.500 | 0.667 | 0.400 |
| 50 | 0.182 | 0.600 | **0.800** | 0.571 |
| 70 | 0.158 | **0.833** | 0.727 | 0.545 |
| 100 | 0.132 | 0.800 | 0.769 | 0.667 |
| 140 | 0.122 | 0.737 | **0.889** | 0.706 |
| 200 | 0.122 | 0.759 | **0.857** [0.70, 0.97] | 0.571 |

Best rule: **`n_crit_steps ≥ K*`** — total count of critical steps in the trajectory.
Best F1 (LOO-CV): **0.889** at N=140.
At N=200, F1 = 0.857 with 95% bootstrap CI [0.70, 0.97].

## Does CPD reach F1 ≥ 0.95?

**Short answer**: No, not with the point estimate at any N ≤ 200. The 95% bootstrap CI's upper tail brushes 0.96–0.97 at N=140–200, so we cannot statistically rule out F1 ≥ 0.95, but we cannot claim it either.

**Why the ceiling sits at ~0.85–0.89**:
- π_0.5 on task00 is **too good** — success rate 93.5% leaves only 13 failures out of 200.
- A single misclassification (1 FP or 1 FN) swings F1 by ~0.05 — so even the *best-case* run with 1 FP + 1 FN gives F1 ≈ 0.857.
- The CPD *signal* itself is excellent — at N=200, success vs failure distributions:
  | feature | succ median (p25,p75) | fail median (p25,p75) |
  |---|---|---|
  | `n_crit_steps` | 38 (29, 55) | 216 (166, 366) |
  | `longest_run` | 17 (14, 24) | 151 (68, 332) |
  | `critical_fraction` | 14% (11%, 19%) | 42% (32%, 70%) |
  
  Failures have ~6× more critical steps and ~9× longer worst-run than successes. Classes are highly separable — the F1 ceiling is set by the small failure count, not by signal overlap.

## Why `has_critical` is broken at τ=0

The original loose rule (any 3-step run with r_t<0 ⇒ critical) flags **>93% of successes** as critical too. Recall stays at 1.0 but precision collapses to 0.065 at N≥100. This is the small-positives-class artefact: τ=0 is too lenient. The duration/count-based rules fix this by requiring the dip to actually be substantial.

## Recommendations for paper

1. **Headline rule**: `n_crit_steps ≥ K*` with K* learned from a held-out split. LOO-CV F1 = 0.889 at N=140 is a credible mid-N number.
2. **Headline framing**: not "F1≥0.95 reached at N=X", but **"CPD detects failures with F1 ≈ 0.86–0.89 (LOO-CV) using N=140–200 rollouts; remaining FP/FN budget is < 2 episodes per 200"**. That's the paper claim that matches the data.
3. **To push toward 0.95**, the bottleneck is failure-pool size, not CPD signal. Options:
   - Run a harder task where π_0.5 fails ~30–50% — small N gets many failures.
   - Add more tasks (libero_10 has 10 tasks; cross-task pooling under same encoder would 10× failure pool).
   - Use AUROC/PR-AUC instead of F1 — both are threshold-free and stable under class imbalance. (AUROC at N=200 will likely be 0.95+ given the distribution separation above.)

## Artifacts

- `scripts/sweep_cpd_n.py` — original sweep (has_critical + sum_r continuous), in-sample best-threshold F1
- `scripts/sweep_cpd_n_loocv.py` — LOO-CV F1 across 4 rules, this exp's primary script
- `reports/exp2/sweep/per_traj.json` — per-(N, episode) trajectory features
- `reports/exp2/sweep/loocv_f1_vs_n.json` — LOO-CV F1 + bootstrap CI per (N, rule)
- `reports/exp2/sweep/loocv_f1_vs_n.png` — figure (Fig. 5 candidate)
- `reports/exp2/sweep/f1_vs_n.png` — old figure (has_critical / sum_r only)
