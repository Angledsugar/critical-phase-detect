# Option B — Action Residual + PPO (main path)

## 한 줄 요약
VLA forward 한 번 → 작은 adapter 가 그 위에 residual action 을 얹음. PPO 로 adapter + value head 만 학습. VLA 내부 weight 는 한 번도 안 건드리고, `src/cpd/policies/ppo_refiner.py` 에 이미 90% 구현되어 있어 wiring + detector gating + eval 만 남음.

---

## 사전 지식 (석사생용)

**Action residual (Residual Policy Learning, Silver 2018; Johannink 2019)**
- Base policy 가 action `a_base` 를 뽑은 뒤, 작은 학습 가능 모듈이 보정량 `Δa` 를 더해 최종 action `a = a_base + Δa` 를 만든다.
- Base 는 frozen — backprop 안 함. 학습 비용은 "작은 모듈" 의 forward + backward 만.
- Zero-init: `Δa = 0` 으로 시작하면 학습 0 step 시점에 정확히 base policy 와 같은 행동.

**왜 LoRA fine-tuning 보다 싼가**
- LoRA 는 transformer 내부 layer 에 W' = W + BA 를 끼워넣음 → forward 와 backward 가 VLA 전체를 통과.
- Action residual 은 VLA forward 1회 (gradient X), 그 결과 위에 작은 MLP 1회. backward 는 그 작은 MLP 만.
- VRAM, step time 모두 LoRA 의 1/5 ~ 1/20.

**PPO (Proximal Policy Optimization, Schulman 2017)**
- on-policy RL. 새 policy 가 old policy 에서 너무 멀리 못 가도록 ratio `ρ_t = π_new / π_old` 를 `[1-ε, 1+ε]` 로 clip.
- 우리는 single trajectory on-policy → 한 rollout 마다 1번 update.

**Detector-gated reward**
- KDE log-ratio reward `r_t = log f̃₊(z_t) − log f̃₋(z_t)` 가 양수인 step = "성공 demo 에 가깝고 실패 demo 에서 먼 state" = critical (positive 방향) phase.
- adapter 가 학습 신호를 critical step 에만 받게 하면 (gating), uninformative step 의 noisy gradient 를 제거 → sample efficiency ↑.

---

## 직관 — 무엇을 하는가

```
LIBERO env --obs--> VLA (frozen) --base_action-->  +  --refined_action--> env
                              |                    ^
                              v                    |
                         AdapterHead --residual----+
                              ^
                              |
                         PPO update <-- detector_gated_reward(z_t) <-- DetectorPipeline
```

VLA 가 행동의 "초안" 을 쓰고, adapter 가 그 초안에 빨간 펜으로 "여기 좀 고쳐" 라고 더한다. 빨간 펜의 잉크 양 (= 학습 신호) 은 detector 가 "지금이 중요한 순간" 이라고 한 step 에만 흐른다.

---

## 학습 환경 — 어떤 LIBERO task 인가

### 1순위 (Week 5-6 main run)

| 항목 | 값 | 출처 |
|---|---|---|
| Suite | `libero_10` (= LIBERO-Long, paper §6.1 main) | `configs/env/libero_long.yaml` |
| Task ID | `0` | exp1/exp2 와 동일 |
| Task instruction | "put both the alphabet soup and the tomato sauce in the basket" | `reports/exp2/summary.md` |
| Image size | 128×128 | `configs/env/libero_long.yaml` |
| Max steps / episode | 600 | `configs/env/libero_long.yaml` |
| Demo 수 (detector 학습용) | 20 (성공 demo, LIBERO 제공 expert hdf5) | `LiberoDemoSource(max_demos=20)` |
| VLA backbone | π_0.5 (`pi05_libero` checkpoint) | `configs/policy/pi0_5.yaml` |
| Baseline 성공률 (refinement 전) | π_0.5 vanilla, 200 ep 기준 **93.5%** | `reports/exp2/summary.md` |
| 잠재 개선 여지 | 6.5% (= 13/200 failure) | exp2 측정 |

### 왜 task00 부터 시작하는가

1. **Sunk cost — encoder + detector 가 이미 이 task 에 학습됨**
   - exp1 의 TLDR encoder, G2 labeler, KDE 통계가 모두 task00 의 200 demo + rollout 으로 fit 되어 있음 (`reports/exp1/summary.md`, `reports/exp2/summary.md`).
   - PPO refiner 만 새로 붙이면 됨 — encoder 재학습 비용 0.

2. **Detector 의 baseline 신뢰도가 측정되어 있음**
   - exp2 LOO-CV: F1 = 0.86–0.89, z-separation = 4.09.
   - "detector 가 어느 정도 맞는지 모르는 상태에서 PPO 를 돌려서 success 안 오를 때 detector 탓인지 PPO 탓인지 모르는" 디버깅 지옥을 회피.

3. **데이터 자산**
   - `/media/engineer/DATA/datasets/cpd_rollouts/pi05_libero/libero_10/task00/` 에 200 episode rollout 이 이미 저장됨 — 초기 PPO warmup 이나 sanity check 에 재활용 가능.

### task00 의 한계 (exp2 가 노출시킨 ceiling)

- **Failure 가 13개뿐** → PPO refinement 가 채울 수 있는 개선폭 자체가 작음 (6.5%p 가 천장).
- **Critical phase 가 비교적 단순** — 두 물체를 순차로 basket 에 넣는 2-stage flow. insertion / fastening 같은 fine-grained motor 부분 적음.
- 이 한계 자체가 **§6.4 ablation 의 motivation** — refinement 의 *상한* 을 task00 에서 측정하고, 헤더운한 task 로 확장.

### 2순위 — Week 6 후반 / Week 7 확장 (선택)

| Suite / task | 추가 이유 |
|---|---|
| `libero_10` task 1-9 | 같은 suite 내 다른 long-horizon task. encoder 재학습 비용 적고 (TLDR transfer), failure pool 10× 확대 → PPO 의 효과 측정 SNR ↑ |
| `libero_object` task 0 | object 변동만 있는 단순 task — refinement 가 *과도하게* fit 안 하는지 sanity. 실패율 낮으면 skip. |
| `libero_goal` task 0 | instruction 만 변동. cross-instruction generalization 의 PPO 영향 확인. |
| (out of scope) `libero_spatial` | position 만 변동 — critical phase 가 거의 없는 task 라 PPO refinement 의 가치 잘 안 드러남. paper Tab 1 에는 넣되 PPO 학습 대상에서 제외. |

> **결정 원칙**: paper §6.1 의 main eval 표 (Tab 1) 는 4 suite 전체. 하지만 PPO 학습은 **`libero_10` task 0 만 main run** 으로, 나머지 task 는 (a) baseline VLA + 학습된 adapter zero-shot transfer 또는 (b) per-task 짧은 fine-tune (50 iter) 로 채움. fine-tune 비용을 §8 timeline 에 가둠.

### 환경 설정 한 줄 요약 (config snippet)

```yaml
# configs/train/ppo_refiner.yaml (env 부분 발췌)
env:
  _target_: cpd.envs.libero.LiberoEnv
  suite: libero_long      # = libero_10
  task_id: 0              # "put both the alphabet soup and the tomato sauce in the basket"
  image_size: 128
  max_steps: 600
demo_source:
  _target_: cpd.envs.libero.LiberoDemoSource
  suite: libero_long
  task_id: 0
  max_demos: 20           # G2 labeler + TLDR encoder 입력
```

---

## 코드 변경 사항

### 새로 만들 파일

| 경로 | 역할 |
|---|---|
| `scripts/train_ppo_refiner.py` | LIBERO env 와 `Pi0RefinedPolicy` 를 wire. CLI: `--suite libero_long --task-id 0 --n-iters 200 --rollouts-per-iter 4 --gating critical_only --ckpt-out ...`. Hydra config 로 `configs/train/ppo_refiner.yaml` 로드. |
| `configs/train/ppo_refiner.yaml` | `policy: ppo_refiner`, `env: libero_long_task00`, `detector: g2_default`, PPO hyperparams (`lr 3e-4, clip_eps 0.2, gamma 0.99, lam 0.95, epochs 4`), `gating.mode: dense\|critical_only\|critical_weighted`, `n_iters: 200`, `rollouts_per_iter: 4`. |
| `scripts/eval_ppo_refiner.py` | 학습된 adapter 로 LIBERO rollout 100회 → success rate. 비교 row: `vla_only`, `vla + dense_residual`, `vla + critical_only`, `vla + critical_weighted`. JSON → `reports/exp_ppo/`. |
| `tests/test_ppo_refiner_gating.py` | gating mode 3개 각각 dtype/shape 검증 + `critical_only` mode 에서 non-critical step 의 gradient = 0 인지 unit test. |

### 수정할 파일

| 경로 | 무엇 |
|---|---|
| `src/cpd/policies/ppo_refiner.py` | `Pi0RefinedPolicy.__init__` 에 `gating: Literal["dense","critical_only","critical_weighted"] = "dense"` 추가. `refine_step` 안에서 `rewards` 에 critical mask 적용 — `critical_only` 는 zero-out, `critical_weighted` 는 reward 자체 (log-ratio) 를 곱. mask 는 `reward_fn.is_critical(latents)` 로 받음. |
| `src/cpd/core/reward.py` | `Reward.is_critical(latents) -> Tensor[bool]` 추가. 1줄: `return self.per_step(latents) > 0`. KDE 통계로 이미 다 계산된 값. |
| `pyproject.toml` | (변경 없음 — 이미 필요한 dep 다 있음) |

### 이미 있는 파일 (재사용)

- `src/cpd/policies/ppo_refiner.py` — `AdapterHead`, `Pi0RefinedPolicy.refine_step` (PPO loop) 그대로 사용.
- `src/cpd/policies/pi0_5.py` — `Pi05Policy` (frozen base) 그대로.
- `src/cpd/envs/libero.py` — `LiberoEnv`, `LiberoDemoSource` 그대로.
- `src/cpd/core/pipeline.py` — `DetectorPipeline.ingest` / `refresh_stats` 그대로.
- `scripts/collect_libero_rollouts.py` — train script 가 import.

---

## paper_plan.md 변경 사항

### §3.5 Reward design — 마지막에 한 단락 추가
> "Detector-gated PPO: adapter residual 의 학습 신호를 critical step (`r_t > 0`) 에만 부여한다. dense / critical_only / critical_weighted 세 mode 를 §6.4 ablation 9 에서 비교."

### §6.1.2 VLA backbone — 표 아래 "Refinement 방식" 소절 신설 (5-6줄)
> "VLA 자체는 frozen. 학습 가능한 부분은 obs → low-rank residual (rank=8, ~수천 param) 의 작은 adapter (`AdapterHead`) 와 value head 뿐. Action 은 `a_t = π_VLA(o_t) + W_up · relu(W_down · z_t)` 로 산출되며, `W_up` 을 0 으로 초기화해 학습 0 step 시점에서 baseline VLA 와 정확히 동일. PPO 가 adapter + value head 만 update — VLA forward 는 한 번만, gradient 없이. 이로써 LoRA fine-tuning 대비 GPU-hour 비용을 ~1 order 절감 (§6.4 ablation 12 에서 정량 비교)."

### §6.4 Ablation 설계 — 9, 10, 11 추가
```
9.  Refinement gating: dense vs critical_only vs critical_weighted        (gating 의 가치)
10. Adapter rank: r ∈ {2, 8, 32}                                          (capacity vs cost)
11. Sample efficiency: success rate vs n_iters 곡선 (gating mode 별)       (gating 의 학습 속도)
```

### §7 Figure — Fig 6 추가
> "Fig 6 (new): success rate vs PPO iter, dense vs critical_only — critical_only 가 같은 success 에 도달하는 데 필요한 iter 수가 작음 (예측)."

### §8 Timeline — Week 5-6 항목에 추가
> "Week 5: PPO refiner wiring (`scripts/train_ppo_refiner.py`) + detector gating + unit test.
> Week 6: LIBERO-Long task 0 training (200 iter × 4 rollout) + eval (100 rollout) → Tab 1 의 π_0.5 row 채움."

---

## Q&A — 석사생 예상 질문

**Q1: VLA 가 잘못된 action 을 뽑으면 residual 로 정말 고쳐지나?**
- "VLA 의 action 이 행동 manifold 의 옳은 영역 *근처에는* 있다" 라는 가정 하에 residual 로 미세 보정 가능. 완전히 다른 영역으로 가야 하는 실패는 못 고침 (Limitation — §10).

**Q2: detector 가 틀려서 non-critical step 을 critical 로 표시하면?**
- false positive 는 noisy gradient 일 뿐 — PPO clip 이 버퍼 역할. False negative (진짜 critical 을 놓침) 가 더 위험 → conf metric (§3.6) fallback 으로 conf 낮으면 dense mode 로 복귀.

**Q3: Adapter rank 가 너무 작으면 capacity 부족 아닌가?**
- §6.4 ablation 10 에서 r ∈ {2, 8, 32} 로 검증. 8 이 default 인 이유는 LoRA 관행 + LIBERO action dim (7) 와 비슷한 order.

**Q4: Single-trajectory PPO 가 unstable 하지 않나?**
- 맞음. 그래서 `clip_eps=0.2`, `epochs=4` 로 보수적이고, value head 도 단일 linear (Linear(in_dim, 1)) 로 가볍게 둠. Multi-trajectory batched PPO 는 §11 Out of Scope.

---

## 작업 일정 / 비용

| 항목 | 일수 | 비고 |
|---|---|---|
| `train_ppo_refiner.py` + config + gating | 2 | gating logic 핵심 |
| `is_critical` + 1 라인 wiring | 0.5 | reward.py 1 method 추가 |
| Unit test | 0.5 | gradient mask 검증 |
| 첫 LIBERO-Long training run | 1-2 | 200 iter × 4 rollout, 1 GPU |
| Eval script + 100 rollout | 1 | success rate 4 row |
| paper_plan.md 추가 | 0.5 | §3.5, §6.1.2, §6.4, §7, §8 |
| **합계** | **5-6일** | Week 5-6 안에 끝남 |

GPU-hour: π_0.5 forward 200 × 4 × ~150 step ≈ 12만 forward. A100 1대로 12-18시간 예상.

---

## 우선순위 / 의존성

1. **선행**: `Reward.is_critical` 추가 (5분 분량) — `train_ppo_refiner.py` 가 이걸 import.
2. **메인 작업**: `train_ppo_refiner.py` + gating mode 3개.
3. **검증**: unit test → 1 task 짧은 run (50 iter) → 정상이면 200 iter full run.
4. **보고**: `reports/exp_ppo/summary.md` 에 success rate 4 row + Fig 6 plot.
5. **paper 반영**: §6.1.2 / §6.4 / Tab 1 갱신.

블로커 — Q3 (RLT supervised label) 해결 안 되어도 Option B 자체는 진행 가능 (success rate 는 LIBERO 내장 reward 로 측정).
