# Critical-Phase Videos — Exp2 (task00 LIBERO-Long, π₀.₅, N=200 LOO)

15 representative episodes from the 200-rollout buffer. KDE is built leave-one-out
(the scored episode is **excluded** from B⁺/B⁻ so it stays out-of-sample), the
remaining 199 trajectories form the buffer.

Per episode you get two artifacts:

- `fig4_<ep>.png` — the r_t curve over time + 6 sampled frames (red border = critical step).
- `<ep>_cp.mp4` — full rollout video at 30 fps, frame border turns **red** when r_t < 0
  for ≥3 consecutive steps (the debounced critical-phase rule from §4 of the paper).

Rule: `critical(t) = (r_t < τ=0) AND part of a run of length ≥ 3`,
where `r_t = log f̃⁺(z_t) − log f̃⁻(z_t)` (Bayes-optimal log-ratio).

## What to look for

| group | episodes | what to expect on screen |
|---|---|---|
| **clean failure** (long, persistent red) | ep068, ep118, ep049, ep099, ep199, ep018, ep015 | red border lights up early and stays on; longest_run ≥ 150 steps. CPD's strongest signal. |
| **borderline failure** (intermittent red) | ep005, ep009, ep134, ep148, ep179, ep189 | red flickers in bursts of 50–100 steps; CPD still flags via `n_crit_steps`, but harder calls. |
| **clean success** (almost no red) | ep115 | only 7 critical steps total (2.65%). CPD correctly says "fine". |
| **recovering success** (lots of red but still succeeds) | ep149 | 63% critical steps, longest_run=281. The robot lingers in a "failing-looking" region then recovers — this is the **F1-ceiling false positive**: CPD flags it because it looks identical to a real failure in latent space. |

## Per-episode numbers (from `per_episode.json`)

| episode | succ | T | n_crit | longest_run | crit_frac | min r_t |
|---|---|---|---|---|---|---|
| ep005 | ✗ | 520 | 230 |  79 | 44.2% |  −7.41 |
| ep009 | ✗ | 520 | 130 |  62 | 25.0% | −16.39 |
| ep015 | ✗ | 520 | 192 | 151 | 36.9% |  −9.46 |
| ep018 | ✗ | 520 | 216 | 185 | 41.5% |  −2.99 |
| ep049 | ✗ | 520 | 388 | 353 | 74.6% |  −2.11 |
| ep068 | ✗ | 520 | 485 | 485 | 93.3% |  −8.79 |
| ep099 | ✗ | 520 | 366 | 332 | 70.4% |  −2.88 |
| **ep115** | **✓** | 264 |   7 |   7 |  **2.7%** |  −0.30 |
| ep118 | ✗ | 520 | 498 | 488 | 95.8% |  −9.53 |
| ep134 | ✗ | 520 | 132 |  52 | 25.4% | −19.48 |
| ep148 | ✗ | 520 | 173 |  52 | 33.3% |  −4.06 |
| **ep149** | **✓** | 514 | 325 | 281 | **63.2%** |  −3.32 |
| ep179 | ✗ | 520 | 166 |  68 | 31.9% |  −1.88 |
| ep189 | ✗ | 520 | 129 |  99 | 24.8% |  −1.71 |
| ep199 | ✗ | 520 | 319 | 212 | 61.4% |  −2.25 |

## Reproduce

```bash
.venv/bin/python scripts/extract_critical_phase.py \
    --out reports/exp2/critical_phase \
    --episodes ep005,ep009,ep015,ep018,ep049,ep068,ep099,ep115,ep118,ep134,ep148,ep149,ep179,ep189,ep199
```

`--episodes` filters which trajectories get rendered. The KDE buffer is always built
from **all 199 other rollouts** in `--rollouts` regardless of this flag — exp2's
N=200 LOO setting is preserved.

## Why these 15

- **13 failures (all of them)** — the entire failure pool for task00 at N=200. These are
  the positives in the F1 calculation; the videos let you verify that "longest_run" /
  "n_crit_steps" really does correspond to visible task failure.
- **ep115** (clean success) — sanity case: a normal successful rollout should have
  nearly no red. It does (2.7%).
- **ep149** (recovering success, FALSE POSITIVE under `crit_fraction ≥ 0.5`) — this is
  the single biggest contributor to the F1 ceiling reported in `summary.md`. Watching
  this one explains *why* F1 plateaus at ~0.86–0.89: the model genuinely behaves like a
  failing rollout for ~280 steps before recovering, and CPD has no way to know that.
