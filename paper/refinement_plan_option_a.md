# Option A — VLA 내부 LoRA Refinement (cost-comparison baseline)

## 한 줄 요약
π_0.5 transformer 내부 layer 에 LoRA adapter 를 끼워넣고 PPO 로 LoRA weight 를 학습. VLA 본체 weight 는 frozen 이지만 forward/backward 가 VLA 전체를 통과하므로 GPU-hour 비용이 Option B 의 5-20배. Paper 에선 main 이 아니라 §6.4 ablation 12 의 **cost vs performance** 비교 row 로만 등장.

---

## 사전 지식 (석사생용)

**LoRA (Low-Rank Adaptation, Hu 2021)**
- Pretrained weight `W ∈ R^{d×k}` 를 직접 안 건드리고, 옆에 작은 path `BA` 를 더함:
  `W' = W + (α/r) · B · A`, with `A ∈ R^{r×k}`, `B ∈ R^{d×r}`, `r ≪ min(d, k)`.
- `B` 는 0 으로 초기화 → 학습 0 step 시점에 `W' = W`.
- 학습 가능한 param 은 `r·(d+k)` 만 — d=4096, k=4096, r=8 이면 65K params (vs 16M of full W).

**왜 비싼가 (Option B 와의 차이)**
- LoRA 도 param 자체는 작지만, **VLA 전체를 forward + backward** 해야 gradient 가 LoRA layer 에 도달.
- π_0.5 같은 multi-billion-param VLA 는 backward 1회만 해도 VRAM 수십 GB, step time 수백 ms.
- Option B (action residual) 는 VLA forward 1회 (no_grad) + 작은 MLP backward → ~1 order 더 쌈.

**peft 라이브러리**
- HuggingFace 의 `peft` 가 `LoraConfig(target_modules=[...], r=8, alpha=16)` 만 지정하면 자동 inject.
- `target_modules` 는 보통 `["q_proj", "v_proj"]` 같은 attention projection. π_0.5 의 정확한 module name 은 openpi 코드 확인 필요.

**왜 그래도 baseline 으로 둘까**
- "Action residual 로 못 고치는 실패 모드" 가 있을 가능성 — VLA action manifold 자체가 잘못된 경우.
- LoRA 는 VLA 내부 representation 을 살짝 바꿀 수 있어, residual 이 reach 못 하는 fix 가 가능할 수 있음.
- 두 방식의 success-rate gap 이 곧 "external residual 의 한계" 를 정량화 → paper Discussion 의 정직한 limitation 진술 재료.

---

## 직관 — 무엇을 하는가

```
LIBERO env --obs--> VLA (frozen weight + LoRA) --action--> env
                              ^                     |
                              |                     v
                              +-- backward gradient -- PPO loss <-- detector_gated_reward
                              (LoRA params 만 update)
```

VLA 의 transformer block 안 곳곳에 작은 "수정 path" 를 끼워두고, RL gradient 가 그 path 의 weight 만 학습. VLA 본체는 freeze 지만, gradient 는 본체를 통과해 흘러야 한다 — 그래서 비싸다.

---

## 코드 변경 사항

### 새로 만들 파일

| 경로 | 역할 |
|---|---|
| `src/cpd/policies/lora_refiner.py` | `Pi0LoRARefinedPolicy(base, reward_fn, config)`. `__init__` 에서 `peft.get_peft_model(base._policy.model, LoraConfig(...))` 로 LoRA inject. `predict` 는 LoRA 활성 forward. `refine_step` 은 PPO loss 를 LoRA param + value head 에 backprop. value head 는 `nn.Linear(hidden_dim, 1)` 신규. |
| `configs/policy/lora_refiner.yaml` | `target_modules: [...]` (π_0.5 module name 결정 후 채움), `r: 8`, `alpha: 16`, `dropout: 0.0`, PPO hyperparams (B 와 동일). |
| `scripts/train_lora_refiner.py` | Option B 의 `train_ppo_refiner.py` 와 골격 동일, policy 만 `Pi0LoRARefinedPolicy` 로 swap. CLI: `--suite libero_long --task-id 0 --n-iters 200 --rollouts-per-iter 4 --lora-r 8 --gating critical_only --ckpt-out ...`. |
| `scripts/eval_lora_refiner.py` | 학습된 LoRA 로 LIBERO rollout 100 회 → success rate. Option B eval 과 같은 metric, JSON → `reports/exp_lora/`. |
| `tests/test_lora_refiner.py` | (1) LoRA inject 후 trainable param count 가 frozen 의 1% 미만인지 assert. (2) 학습 0 step 시점에서 base policy action 과 정확히 일치 (B=0 init 검증). (3) gating mode 3 개 동작. |

### 수정할 파일

| 경로 | 무엇 |
|---|---|
| `pyproject.toml` | optional dep 추가: `[project.optional-dependencies]` 의 `vla` extras 에 `peft>=0.10` 추가, 또는 별도 `lora` extras 신설. |
| `src/cpd/policies/base.py` | (변경 없음 — `Pi0LoRARefinedPolicy` 가 `predict` + `refine_step` Protocol 그대로 따름) |
| `src/cpd/core/reward.py` | (Option B 가 이미 `is_critical` 추가했다는 가정 — 안 되어 있으면 동일하게 1 라인 추가) |

### 새로 안 만드는 것

- LIBERO env wrapper, DemoSource, DetectorPipeline — 그대로 재사용.
- π_0.5 frozen wrapper (`Pi05Policy`) — 그대로 재사용. 단, `lora_refiner` 는 `Pi05Policy._policy.model` 에 직접 LoRA inject (openpi 의 model 객체에 접근).

---

## paper_plan.md 변경 사항

### §3.5 VLA RL Refinement (related work) — "우리 차별" 단락 다음에 1줄 추가
> "구현 측면에서 우리는 외부 action residual 을 default 로 두고, 내부 LoRA refinement 는 §6.4 ablation 12 의 cost baseline 으로만 비교한다 — 동일 detector-gated reward 아래 두 refinement strategy 의 cost / performance trade-off 를 분리 측정."

### §6.3 Metrics — "Cost" 행 신설
| 종류 | metric | 측정 대상 |
|---|---|---|
| **Refinement cost** | GPU-hour per training run, trainable parameter count | external action residual ≪ internal LoRA 의 정량 비교 |

### §6.4 Ablation 설계 — 12 추가
```
12. Refinement strategy: external action residual (Pi0RefinedPolicy)
                       vs internal LoRA          (Pi0LoRARefinedPolicy)
    — same detector_gated_reward, same n_iters; report
      (success rate, GPU-hour, trainable param count, peak VRAM)
```

### §10 Risks & Mitigations — 새 risk 추가
> "Risk: action residual 만으로는 VLA action manifold 가 잘못된 영역에 있는 실패 모드를 수정할 수 없다.
> Mitigation: §6.4 ablation 12 가 internal-LoRA vs external-residual 의 success-rate gap 을 정량화. gap 이 의미 있게 크면 conclusion 에서 future work 로 명시 (예: 'high-stakes precision phase 에선 internal LoRA 가 필요할 수 있다')."

### §11 Out of Scope — 하나 명시
> "본 paper 는 internal LoRA 를 main result 로 제출하지 않는다 — Tab 1 의 모든 row 는 external action residual 기반. Internal LoRA 는 §6.4 의 cost-comparison ablation 으로만 등장."

### §7 Figure — 추가 안 함 (table 만으로 충분)

---

## Q&A — 석사생 예상 질문

**Q1: `target_modules` 를 어떻게 정하나?**
- π_0.5 의 backbone 이 무엇인지 (PaliGemma + diffusion head?) openpi 코드 확인. 일반적으로 attention 의 `q_proj`, `v_proj` (간혹 `o_proj`) 부터 시도. cross-attention 이 instruction conditioning 에 결정적이면 거기 우선.

**Q2: 왜 `r=8` 인가?**
- LoRA 관행. `r=4` 는 너무 capacity 부족할 수 있고 `r=16+` 는 비용 ↑ 로 baseline 의미 약화. 본 paper 에선 cost-comparison 이 목적이라 capacity sweep 은 안 함.

**Q3: VRAM 이 모자라면?**
- gradient checkpointing 으로 memory ↓ time ↑ trade-off. 그래도 안 되면 batch_size=1 + accumulation. peft 가 8-bit / 4-bit base 도 지원하지만 본 paper 의 cost-comparison 은 fp16 기준으로 통일하는 게 비교 공정.

**Q4: External residual 이 항상 이기는데도 이 ablation 이 의미 있나?**
- 두 가지 의미: (1) "왜 우리는 external 을 골랐는가" 의 정량 근거. (2) "어디서 external 이 깨지는가" 의 task-level breakdown — task by task 로 보면 LoRA 가 이기는 task 가 있을 수 있고, 그게 future work 의 시드.

**Q5: 학습 안정성 차이?**
- LoRA 가 더 큰 capacity → variance 도 큼. PPO clip_eps 를 0.1 (B 보다 작게) 로 잡거나 lr 도 1e-4 로 내림. config 에 명시.

---

## 작업 일정 / 비용

| 항목 | 일수 | 비고 |
|---|---|---|
| openpi 안에서 π_0.5 model 구조 파악 + target_modules 결정 | 1 | 가장 큰 미지수 |
| `lora_refiner.py` 구현 (peft inject + value head + PPO step) | 2 | B 의 PPO loop 이식 |
| `train_lora_refiner.py` + config | 1 | B script 의 fork |
| Unit test (param count, init equivalence) | 0.5 | |
| 첫 LIBERO-Long training run | 3-5 | B 의 5-20배. A100 1대로 60-200시간 가능 |
| Eval + cost 측정 | 1 | GPU-hour, VRAM, success rate |
| paper_plan.md 추가 | 0.5 | §3.5, §6.3, §6.4, §10, §11 |
| **합계** | **9-12일** | Option B 끝난 뒤 Week 7+ |

GPU-hour: π_0.5 forward + backward 200 × 4 × ~150 step. A100 1대로 60-200시간 (정확 측정이 ablation 의 본질).

---

## 우선순위 / 의존성

1. **Option B 가 먼저 작동해야 함** — `train_ppo_refiner.py` 의 PPO loop, gating logic, eval pipeline 을 그대로 재사용하기 위함.
2. **선행 조사**: openpi 안의 π_0.5 model structure (`type(Pi05Policy._policy.model)`, parameter name pattern) — 1일 분량의 spike.
3. **메인 작업**: `lora_refiner.py` + train script + eval.
4. **검증**: 0-step 에서 base policy 와 동일 action — 이게 안 맞으면 LoRA inject 가 잘못된 것.
5. **보고**: `reports/exp_lora/summary.md` 에 cost row + success rate gap.
6. **paper 반영**: §6.4 Tab 2 (또는 신규 Tab 3) 에 비교 row.

블로커 — π_0.5 의 model 객체에 `peft.get_peft_model` 이 안 붙으면 (구조가 standard transformers 와 다르면) custom adapter wrapping 필요 → 일정 +3-5일.

---

## Option B 와의 비교 (한 표 요약)

| 항목 | Option B (action residual) | Option A (internal LoRA) |
|---|---|---|
| VLA forward gradient | X (no_grad) | O (full backward) |
| Trainable param | ~수천 (rank=8 adapter) | ~수만 (LoRA r=8 × N layers) |
| GPU-hour / 200-iter run | 12-18h | 60-200h |
| Peak VRAM | ~8GB | ~40GB+ |
| 0-step = base policy | 보장 (W_up=0) | 보장 (B=0) |
| Action manifold 자체 수정 | 불가 | 가능 (잠재) |
| Paper 위치 | Tab 1 main | §6.4 ablation 12 cost row |
| 일정 | Week 5-6 (5-6일) | Week 7+ (9-12일) |
