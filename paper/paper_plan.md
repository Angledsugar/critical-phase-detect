# Paper Plan — G2 기반 Critical Phase Detection

> **목적**: 논문 작성 관점의 master document. 구현·식 detail 은 `plan.md`, 대안 옵션은 `reports/alternatives.md` 참조.
> **Scope**: Goal 정의 G2 (Temporal-Distance Embedding), state $= \varphi(s_t)$, kernel-weighted buffer statistics.
> **Audience**: 본 문서는 석사 1-2 학년 학생이 처음 읽고도 추가 질문 없이 paper 의 골격·용어·실험 의도를 이해할 수 있도록 작성됨.

---

## 0. 사전 지식 / 용어 정리

본 plan 의 §1 부터를 읽기 전 반드시 알아야 할 개념들. 이미 익숙하면 §1 로 이동.

### 0.1 핵심 약자 / 모델 이름

| 약자 | 풀어쓰기 | 한 줄 설명 |
|---|---|---|
| **VLA** | Vision-Language-Action model | 카메라 이미지 + 자연어 명령을 입력으로 받아 로봇의 action 을 출력하는 거대 모델. 예: π_0, RT-2, OpenVLA. |
| **RL** | Reinforcement Learning | reward signal 을 최대화하도록 policy 를 학습하는 패러다임. |
| **GCRL** | Goal-Conditioned RL | "목표 $g$ 가 주어졌을 때 $g$ 에 도달하라" 는 RL setting. |
| **HER** | Hindsight Experience Replay (Andrychowicz+ 2017) | 실패한 trajectory 를 "도착한 곳을 goal 로 재해석" 해서 학습 데이터로 재활용하는 기법. |
| **TLDR** | Temporal-Distance Latent Representation | 두 state 의 latent 거리가 "몇 step 떨어져 있는가" 와 같아지도록 학습한 embedding. |
| **RLT** | RL Token (handoff predictor 논문, 2026) | VLA trajectory 의 어디서부터 RL 에 넘기면 좋을지를 supervised classifier 로 예측. **본 paper 의 직접 비교 대상**. |
| **LIBERO** | Liu+ 2023 manipulation benchmark | 4 suite 로 구성: Spatial / Object / Goal / Long. |
| **π_0, π_0.5** | Physical Intelligence VLA | 본 paper 가 사용하는 두 VLA backbone (둘 다 main, §6.1.2). |
| **KDE** | Kernel Density Estimation | 점 데이터들로 확률밀도를 추정하는 비모수 (non-parametric) 방법. |

### 0.2 자주 쓰는 단어

- **Trajectory $\tau$**: state-action 의 시퀀스 $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$. 한 episode 의 전체 기록.
- **Buffer**: trajectory 들을 모아둔 저장소. 본 framework 에서는 두 개 — 성공 buffer $B_+$, 실패 buffer $B_-$.
- **Demo (demonstration)**: 전문가 (또는 잘 작동하는 VLA) 가 만든 성공 trajectory. 본 plan 에서 task 당 $N \in \{5, \ldots, 20\}$ 개만 사용 (few-shot).
- **Cell $c_t$**: state $s_t$ 를 어떤 식별자(ID 또는 vector) 로 매핑한 것. 본 plan 의 cell 정의 = $\varphi(s_t)$ (latent vector).
- **Latent / embedding**: state 를 NN 으로 변환한 저차원 표현 ($\mathbb{R}^d$, $d \in \{64, 128\}$).
- **Critical phase**: trajectory 안에서 "여기서 실수하면 task 전체가 망가진다" 는 결정적 구간. 예: peg insertion 의 마지막 정렬, screw fastening 의 회전 시작 직전.
- **Self-supervised**: 사람 라벨 없이, 데이터 자체의 구조에서 학습 신호를 만드는 학습 방식.
- **Few-shot**: 새 task 에 적은 (5-20 개 수준) 데이터로 적응.

### 0.3 수학 사전

- **Lipschitz 연속 ($L$-Lipschitz)**: 함수가 너무 급격하게 변하지 않는다는 성질. 식: $\|\varphi(s_a) - \varphi(s_b)\| \leq L \cdot d(s_a, s_b)$. 직관: 입력이 조금 바뀌면 출력도 조금만 바뀐다.
- **Quantile (분위수)**: 정렬된 데이터의 특정 비율 위치 값. 예: 95% quantile = 95번째 백분위 값. 본 plan 에서 $\varepsilon$ 는 demo distance 의 quantile.
- **Kernel density (KDE) — 직관**: continuous space 에서 데이터 점이 직접 같은 위치에 또 등장할 확률은 0. 따라서 각 점 주위에 작은 종 모양 (Gaussian kernel) 을 얹어 합산 → 부드러운 밀도 추정. 본 plan 의 $\tilde{f}_\pm$ 가 이것.
- **Bandwidth $h$**: kernel 의 폭. 작으면 sharp, 크면 smooth. 본 plan 은 $h$ 를 buffer 통계에서 자동 도출 (Silverman's rule).
- **Cover ($\rho$-cover)**: 점 집합이 어떤 영역을 "$\rho$ 이내로 모두 덮는다" 는 의미. 즉 영역 안 임의의 점에서 가장 가까운 데이터까지 거리가 $\rho$ 이하.

### 0.4 본 paper 가 풀려는 문제 — 한 단락

VLA 가 manipulation task 를 수행 중 어딘가에서 실패한다. 이 "실패가 시작되는 critical phase" 를 자동으로 찾아내고, 거기에 RL fine-tuning 을 집중하면 효과적이다 (RLT, 2026 입증). 그러나 critical phase 를 찾는 기존 방법은 (i) 사람 라벨 (RLT) 또는 (ii) 환경 designer 가 코딩한 oracle predicate 가 필요하다 — task 마다 비용 발생. **본 paper 는 demo 5-20 개의 마지막 state 만으로 critical phase detector 를 자동 구성한다**. 사람 라벨도, 술어 코딩도 없다. 이 detector 가 만드는 reward 로 VLA 를 RL refinement.

### 0.5 내부 라벨 crosswalk (alternatives.md / plan.md 와 cross-reference)

본 plan 은 plan.md, reports/alternatives.md 에서 사용하는 옵션 라벨을 일부 인용. 외부 표현 (§2 contributions, §6 baseline 등) 에서는 풀어 쓰지만, 내부 의사결정 추적 (§9 Q-list, §11 out of scope, §12 Theorem statement) 에서는 짧은 라벨 유지.

| 라벨 | 풀어쓰기 |
|---|---|
| **G1** | 환경 designer 가 작성한 oracle predicate (성공/실패 함수) |
| **G2** | 본 paper 가 채택한 self-supervised labeler (φ-distance 기반, demo 통계로 자동 구성) |
| **G3** | VLA instruction + classifier 라벨러 (out of scope, alternatives.md) |
| **G4** | 외부 임의 라벨러 — framework 일반성 추상화 (plug-in 가능, alternatives.md) |
| **C1** | state 기반 cell ($c_t = s_t$) — 본 paper 채택 |
| **C3** | state-action joint cell — out of scope, alternatives.md |

---

## 1. Paper 정체성

| 항목 | 값 |
|---|---|
| Working title (안) | "Self-Supervised Critical Phase Detection for VLA Refinement via Temporal-Distance Goals" |
| Tagline | "VLA 가 흔들리는 순간을, 사람의 라벨 없이 detect 한다" |
| Target venue | **CoRL 2026** |
| Deadline | CoRL 2026 일정 확인 필요 (통상 6-7월 abstract / 9월 full paper) |
| Length | 8 pages + appendix |
| Evaluation | LIBERO benchmark (4 suite, Long 1순위) + π_0 + π_0.5 (둘 다 main) |

---

## 2. Core Contributions (claim)

본 paper 의 기여는 **하나의 강한 main contribution + 그 framework 가 자연스럽게 만족하는 두 가지 desirable property**. 3 개 병렬이 아닌 1 + 2 구조로 제시.

### 2.1 Main Contribution — Self-Supervised Critical Phase Detector for VLA Refinement

**무엇**: VLA trajectory 안의 critical phase (실패가 시작되는 결정적 구간) 를, 사람의 라벨이나 환경 designer 가 작성한 술어 (predicate) 없이, demo 5-20 개만으로 자동 검출하는 framework. 이 detector 가 만드는 reward 로 VLA 를 RL fine-tune.

**어떻게** — 두 단계로 구성:
1. **Self-supervised trajectory labeler**: demo 들의 마지막 state $s_T^{(i)}$ 를 temporal-distance embedding $\varphi$ 로 매핑한 평균을 goal 점 $g$, demo distance 의 quantile 을 임계 $\varepsilon$ 로 정의. 새 trajectory 의 $\|\varphi(s_T) - g\|$ 가 $\varepsilon$ 미만이면 성공, 아니면 실패. **$g, \varepsilon$ 모두 demo 통계에서 자동 도출** — 사람·술어 개입 0.
2. **Buffer-statistic reward**: 라벨링된 trajectory 들로 buffer $B_+, B_-$ 구성. 각 state 가 두 buffer 에 등장하는 kernel-weighted 빈도 차이를 reward 로 변환. 이 reward 로 RL refinement.

**왜 중요**: 기존 critical phase detector 는 (i) 사람이 trajectory 마다 critical 시점을 라벨하거나 (RLT) (ii) 환경 designer 가 task 마다 oracle predicate 를 코딩 (예: "컵이 손에 잡혔다") 해야 했다. 둘 다 task-by-task 비용. 본 framework 는 demo 만 있으면 자동 reconfigure → VLA 의 multi-task 정신과 정합.

> **Reviewer 가 의심할 핵심 질문**: "이 self-supervised 라벨러가 환경 designer 가 만든 oracle predicate 와 정말 일치하는가?" → §6.4 ablation 1 에서 일치율 측정 (목표 ≥ 90%) + §12 Theorem 1 로 점근적 일치 보장.

### 2.2 Desirable Properties of the Framework

위 main contribution 이 다음 두 성질을 자연스럽게 만족 — 본 paper 의 핵심 차별점이자 sub-claim.

#### Property 1 — 외부 Hyperparameter 0

Reward 함수의 모든 hyperparameter (kernel bandwidth $h$, 임계 $\varepsilon$, conf threshold, trajectory weight) 가 **buffer/demo 통계에서 자동 결정**. 사용자가 tuning 할 항목 0.

- 예: $\varepsilon$ = demo distance 의 95% quantile, $h$ = Silverman's rule, conf threshold = buffer occupancy 비율.
- **왜 중요**: hyperparameter tuning 은 RL 논문의 가장 큰 reproducibility 위협. 본 framework 의 robustness 와 cross-task transfer 가능성에 직결.

#### Property 2 — Few-Shot Transferability

새 task 에 **demo 5-20 개만 추가하면 detector pipeline 이 자동 재구성**. predicate rewriting, classifier retraining 모두 불필요.

- $\varphi$ encoder 는 task-agnostic 으로 사전훈련 (TLDR loss). $g, \varepsilon$, buffer statistic 만 새 demo 로 갱신.
- **왜 중요**: VLA 의 가치는 multi-task generalization. 기존 critical phase detector 들은 single-task. 본 접근이 VLA pipeline 과 자연스럽게 결합.

> **Reviewer 가 의심할 추가 질문**: "외부 hyperparameter 가 정말 0 인가?" → 모든 통계 도출식을 §3 표로 명시화 + §12 Theorem 1 의 가정으로 정당화.

---

## 3. Related Work Positioning

각 영역을 한 문단으로 — "그 논문이 무엇을 했고, 우리는 무엇이 다른가".

### 3.1 Critical Phase Detection — 직접 경쟁

**RLT (RL Token, 2026)**: VLA trajectory 에서 "어디서부터 RL 로 fine-tune 할지 (handoff 시점)" 를 supervised classifier 로 예측. 사람이 trajectory 마다 handoff 시점을 라벨링.

**우리 차별**: RLT 는 supervised — 라벨 비용이 task-by-task. 우리는 demo 만으로 unsupervised. F1 비교 (§6.3) 에서 **우리의 unsupervised 라벨이 RLT 의 supervised 라벨에 얼마나 근접하는가** 가 핵심 지표.

### 3.2 Temporal-Distance Representations — encoder 기반

- **TLDR (Temporal-Distance Latent Representation)**: state 쌍의 latent 거리가 step distance (몇 step 떨어져 있나) 와 같아지도록 contrastive 학습. 본 paper 의 default $\varphi$.
- **QRL (Quasi-metric RL)**: 비대칭 거리 — "$s_a$ 에서 $s_b$" 와 "$s_b$ 에서 $s_a$" 가 다름. irreversible task (예: 깨지는 물체) 에 적합.
- **HILP (Hilbert Representations)**: Hilbert space 기반, inner product/metric 모두 정의.

**우리 차별**: 우리는 $\varphi$ 자체를 새로 제안하지 않는다. TLDR 을 그대로 채택, **critical phase detection 으로 응용 확장**. QRL/HILP 는 §6.4 ablation 2 에서 $\varphi$ 선택의 영향만 비교.

### 3.3 Goal-Conditioned RL — 이론 배경

- **HER (Hindsight Experience Replay)**: 실패 trajectory 를 "도착한 곳을 goal 로 재해석" 해서 학습 데이터로 재활용.
- **UVFA (Universal Value Function Approximator)**: $V(s, g)$ 처럼 goal 을 입력으로 받는 value function. multi-goal RL 의 표준.

**우리 위치**: HER 의 사상 ("도달한 곳이 demo goal 근처면 성공") 을 **detector 영역으로 가져옴**. 즉 우리 framework 는 HER 의 GCRL 가정을 critical phase detection 에 응용.

### 3.4 Goal Generation — 관련하지만 비교 X

**GoalGAN, Skew-Fit, MEGA**: 학습용 goal 을 자동 생성 (curriculum learning).

**우리 차별**: 우리는 goal 을 *생성* 하지 않는다. demo 마지막 state 를 그대로 사용. 따라서 직접 비교 대상이 아니라 related work 에서 mention 정도.

### 3.5 VLA RL Refinement — downstream 활용

**π_0, OpenVLA, RT-2 + RL**: 사전훈련된 VLA 를 RL 로 fine-tune.

**우리 차별**: 우리는 VLA 자체를 만들지 않는다. 기존 VLA 에 detector 를 plug-in 해 RL refinement 의 "어디를 학습할지" 를 자동 결정.

### 3.6 Intro 의 narrative 한 단락 (작성 hint)

> "VLA 가 일반 task 에는 잘 작동하지만 정밀 phase (insertion, fastening) 에서 자주 실패. 이를 detect 해서 RL refinement 의 trigger 로 쓰면 효과적임 [RLT, 2026]. 그러나 기존 detection 은 (i) 사람 라벨 [RLT] 또는 (ii) 환경 designer 가 작성한 oracle predicate 에 의존 — 둘 다 task-by-task 비용. 본 paper 는 demo 의 temporal-distance embedding 만으로 critical phase 를 self-supervised 검출, RL refinement 의 reward 로 직결."

---

## 4. Paper 구조 Outline

### 4.1 Section-by-section

| § | 제목 | 페이지 | 핵심 메시지 |
|---|---|---|---|
| 1 | Introduction | 1.0 | Critical phase detection 의 자동화 필요성 + G2 의 한 줄 요약. |
| 2 | Related Work | 0.5 | RLT, TLDR, HER 위치잡기. |
| 3 | Method | 2.5 | Self-supervised labeler + φ encoder + kernel-weighted reward. |
| 4 | Theory | 0.5 | Theorem 1 (G2 ↔ G1 consistency) + 가정 진술. (§12 detail) |
| 5 | Experiments | 2.5 | LIBERO 4 suite + π_0 + π_0.5 (둘 다 main) + ablation. |
| 6 | Discussion / Limitations | 0.5 | OOD φ, demo quality 의존, 이론 가정의 실험적 검증. |
| 7 | Conclusion | 0.25 | 한 단락. |
| - | Appendix | 4-6 | Implementation detail, theorem proof, dataset list. |

### 4.2 Method § 의 sub-구조 (§3 detail)

```
3.1  Problem setup            (state, action, trajectory, buffer 정의)
3.2  G2 goal definition       (φ 임베딩, g 의 demo 평균, ε 의 quantile)
3.3  φ pre-training           (TLDR alignment + uniformity loss)
3.4  Kernel-weighted statistics (f̃_+, f̃_-, bandwidth h 도출)
3.5  Reward design            (per-step A: 빈도, trajectory B: 성공/실패 가중)
3.6  OOD reliability          (conf metric, fallback)
3.7  Algorithm summary        (pseudocode)
```

각 sub-section 의 식은 `plan.md` §2-§5 에 그대로 인용.

---

## 5. Method Narrative — G2 중심으로 어떻게 풀까

### 5.1 핵심 통찰을 어디서 강조할까

| 통찰 | Paper 내 위치 |
|---|---|
| "Goal labeler 자동화" | Intro · Method §3.2 첫 문단 |
| "외부 hyperparameter 0" | Method §3.4 와 §3.6 의 '도출' 식 박스 + Theorem 1 |
| "Few-shot demo 로 transfer" | Method §3.2 마지막 + Experiments transfer ablation |
| "Buffer-statistic reward 의 framework 일반성 (외부 라벨러 plug-in 가능)" | Method §3.7 의 framework 정의 + Discussion |

### 5.2 식의 priority (paper space limited)

**본문 박스 식**:
- $g, \varepsilon$ 도출식 (G2 핵심) — "goal 점 = demo 마지막 state 의 평균, 임계 = demo distance 의 95% quantile"
- $\mathcal{L}_\varphi$ alignment (TLDR 핵심) — "step distance 를 latent distance 에 회귀"
- $\tilde{f}_\pm$ kernel-weighted 통계 — "각 state 가 성공/실패 buffer 에 얼마나 많이 등장했는가의 부드러운 추정"
- Reward $r_t = r^{(A)} + R$ 통합 — "per-step + trajectory-level 결합"

**Appendix 로 빼는 것**:
- $\mathcal{L}_\text{uniform}$ (collapse 방지 옵션 term)
- conf metric 상세 도출
- $h$ derivation (Silverman's rule 유도)
- Pseudocode 전문

### 5.3 직관 Q&A — 석사생용

**Q1: 왜 "마지막 state 만" 으로 trajectory 의 성공을 정의하나?**
- VLA 는 instruction 을 따라 task 를 끝낸다. 끝났을 때 어디 있는지가 곧 "성공/실패" 의 정의. 중간 state 는 다양해도 됨 (multiple valid paths).

**Q2: 왜 step distance 로 $\varphi$ 를 학습하나?**
- $\varphi$ 가 step distance 정보를 보존하면, latent 거리가 곧 "goal 까지 몇 step 남았나" 의 추정치가 된다. 이것이 reward signal 로 자연스럽게 변환됨.

**Q3: 왜 kernel density 가 필요하나?**
- buffer 에 등장한 정확히 같은 state 가 새 trajectory 에 또 나타날 확률은 continuous space 에서 0. kernel 로 "주변 state 들" 까지 부드럽게 가중하면 unseen state 에도 reward 가 정의됨. 즉 generalization 의 수학적 도구.

**Q4: 왜 reward 를 A (per-step) 와 B (trajectory) 두 개로 나누나?**
- Reward A 는 "이 순간 어떤 cell 에 있는가" 의 즉시 신호. Reward B 는 "trajectory 전체가 성공인가" 의 sparse 신호. 둘을 합치면 dense (A) + 정확한 끝 (B) 의 균형이 잡힘.

**Q5: Few-shot demo 5-20 개로 정말 충분한가?**
- §6.4 ablation 6 에서 $N \in \{5, 10, 20\}$ 변동으로 검증. 또한 Theorem 1 이 "$N$ 이 늘면 G1 일치율이 1 로 수렴" 을 증명하므로 한계와 보장이 함께 제시됨.

---

## 6. Experiments 설계

### 6.1 Sim 환경 — LIBERO

본 paper 의 표준 evaluation = **LIBERO benchmark** (Liu et al. 2023). 4 suite 모두 사용. **LIBERO-Long 을 1 순위 main** 으로 잡고, 나머지 3 suite (Spatial / Object / Goal) 순차 확장.

#### 6.1.1 LIBERO 4 suite — 무엇이 다른가

| Suite | 변동 차원 | task 예시 (단순화) | 우리 setting 에서의 역할 |
|---|---|---|---|
| **LIBERO-Spatial** | 같은 물체, 위치만 변동 | "pick up the block" — block 이 다양한 위치 | $\varphi$ 의 position-invariance 검증 |
| **LIBERO-Object** | 다른 물체, 같은 task structure | "pick up the X" 의 X 가 변동 (cup, bowl, plate) | $\varphi$ 의 object-invariance 검증 |
| **LIBERO-Goal** | 같은 scene, 다른 instruction | scene 그대로, instruction 만 다름 | 다중 task 환경에서 G2 라벨러 작동 확인 |
| **LIBERO-Long** | long-horizon, multi-stage | 여러 sub-task 가 연속 (예: open drawer → pick → place → close) | **Main**: critical phase 가 명확히 분리됨 → detector 의 가치 입증에 가장 적합 |

> **Long 을 main 으로 고른 이유**: critical phase 는 stage 사이의 전이 (예: 정렬 → 삽입) 에서 두드러진다. Long suite 는 stage 가 많아 critical phase 빈도와 다양성이 높다.

#### 6.1.2 VLA backbone — π_0 와 π_0.5 (둘 다 main, **Q6 = (a)**)

| VLA | 역할 | 비고 |
|---|---|---|
| **π_0** (Physical Intelligence, 2024) | Main result table 의 첫 row. 자체 detector 학습 + RL refinement 까지 전체 pipeline 수행. | 1 차 backbone |
| **π_0.5** (Physical Intelligence, 2025) | Main result table 의 둘째 row. 별도로 detector 학습 + RL refinement 반복 (full pipeline). | π_0 후속 모델 — 더 큰 scale + 향상된 instruction following |

> **둘 다 main 으로 보이는 이유**: backbone 두 개 모두에서 framework 가 작동하면 "detector 가 특정 backbone 에 우연히 맞춰진 게 아니다" 는 backbone-agnostic 주장이 main result 로 가능. 단점은 실험량 doubled — Week 4 (π_0) + Week 5 (π_0.5) 로 일정 분배 (§8).

> **추가**: §6.4 ablation 8 은 별도로 "한 backbone 으로 학습한 detector 를 다른 backbone 의 rollout 에 적용" — 즉 cross-backbone transfer 도 측정.

#### 6.1.3 Demo source

LIBERO 가 제공하는 expert demo 중 task 당 5-20 trajectory 를 sample → §3 의 G2 라벨러 입력. demo $N$ 은 §6.4 ablation 6 에서 $\{5, 10, 20\}$ 변동.

### 6.2 Baseline

| Baseline | 의미 (석사생용 풀이) | 기대 동작 |
|---|---|---|
| **Oracle predicate** | LIBERO 가 내장한 task success function 을 그대로 라벨로 사용. designer 가 짠 "정답" 역할. | 우리 self-supervised labeler 가 oracle 과 얼마나 일치하나 (Theorem 1 의 실험 버전). |
| **Random-cell** | 임의 state 를 critical 이라 라벨. 무작위. | chance level — sanity floor. |
| **VLA-only (no detector)** | detector 안 쓰고 vanilla VLA 그대로. | downstream 성능의 lower bound. |
| **Frequency only ($r^{(A)}$ only)** | Reward A 만 사용 (per-step 빈도). B 끔. | reward B 의 기여도 분리 측정. |
| **Trajectory only ($r^{(B)}$ only)** | Reward B 만 사용 (trajectory-level). A 끔. | reward A 의 기여도 분리 측정. |
| **RLT supervised handoff** | 비교 SOTA — 사람 라벨로 학습한 supervised classifier. | 우리 unsupervised 가 이 supervised 정확도에 얼마나 근접하나. |

### 6.3 Metrics

| 종류 | metric | 측정 대상 |
|---|---|---|
| **Detection 정확도** | F1 vs **RLT supervised label** (per-step critical phase ground truth), recall@95 precision | Self-supervised labeler 의 신뢰도 (Q4 = RLT label) |
| **Downstream 성능** | π_0 / π_0.5 RL refinement 의 task success rate (LIBERO 4 suite) | detector 의 utility |
| **Transfer** | 새 task 의 detector F1 (cross-task), cross-VLA F1 (§6.4 #8) | few-shot 적응 능력 |
| **OOD 안전성** | conf < threshold 시 fallback rate | §3.6 conf metric 작동 |
| **Hyperparameter robustness** | demo $N$ 변동에 대한 sensitivity | "외부 hyperparameter 0" claim 보강 |

> **Q4 의 결정 (RLT label = ground truth) 의미**: F1 의 정답지가 환경 술어 (G1) 가 아니라 RLT 사람 라벨. 즉 "critical phase 는 RLT 가 정의한 것과 같은 것" 이 본 paper 의 operational definition. G1 oracle 은 보조 baseline 으로만 등장.

### 6.4 Ablation 설계

```
1. Self-supervised labeler vs oracle predicate     (라벨러 신뢰도, Theorem 1 의 empirical 검증)
2. φ encoder: TLDR vs QRL vs HILP vs VLA-backbone    (encoder 선택의 영향)
3. State 표현: Option (b) vs (a) discrete             (kernel-weighted 의 가치)
4. λ ∈ {0, 1} (alignment-only vs +uniformity)         (collapse term 필요성)
5. Reward 분해: A only vs B only vs A+B               (reward 설계 검증)
6. Demo count N ∈ {5, 10, 20}                         (few-shot 한계)
7. Cross-suite transfer matrix on LIBERO (4×4: Spatial/Object/Goal/Long)   (transfer 강건성)
8. Cross-VLA transfer: π_0 demo 로 학습한 detector 를 π_0.5 rollout 에 적용 (또는 역방향)   (cross-backbone 강건성)
```

> **Ablation 8 의 main 과의 차이**: §6.1.2 main 에서는 두 VLA 가 *각자* detector 를 가짐. Ablation 8 은 *한쪽* detector 를 *다른 쪽* 에 적용. 즉 main = parallel pipelines, ablation = transfer.

---

## 7. 주요 Figure / Table 후보

| 자리 | 내용 | 형식 |
|---|---|---|
| Fig 1 | Method overview ($\varphi$, $g$, $\varepsilon$, detector flow) | architecture diagram |
| Fig 2 | TLDR sanity check — step distance vs latent distance scatter (Theorem 1 의 가정 A1 검증) | scatter |
| Fig 3 | Kernel-weighted $\tilde{f}_\pm$ heatmap on 2D toy | heatmap |
| Fig 4 | Critical phase detection trajectory visualization (성공/실패 demo + detector flag) | trajectory plot |
| Fig 5 | Cross-suite transfer matrix (4×4 LIBERO suite, F1 color) | matrix heatmap |
| Tab 1 | Main result — F1 / success rate, π_0 + π_0.5 두 row × 4 suite column | table |
| Tab 2 | Ablation — encoder · state rep · reward 분해 | table |
| Tab 3 | Demo $N$ sensitivity | table |

---

## 8. Timeline / Milestones

> 모든 일정은 CoRL 2026 deadline 확인 후 역산. 아래는 generic 8-week sprint 가정.

| Week | 작업 |
|---|---|
| 1 | LIBERO 4 suite 셋업, π_0 / π_0.5 inference 환경 구축, demo 수집 (LIBERO-Long 우선). RLT supervised label set 확보 또는 자체 라벨 수집 결정 (Q3). |
| 2 | $\varphi$ TLDR pre-training + sanity check (§3.1.4 — Fig 2 의 데이터). LIBERO-Long 에서 G2 라벨러 작동 확인. |
| 3 | Buffer 통계 + reward A/B 구현 + RL detector loop (π_0 backbone). |
| 4 | LIBERO-Long main 결과 (π_0). G1 oracle baseline · RLT supervised baseline 측정. |
| 5 | π_0.5 backbone 으로 동일 pipeline 반복 → main result 의 둘째 row 확보. |
| 6 | Cross-suite transfer (LIBERO-Spatial / Object / Goal). 추가 ablation (encoder · state rep · λ · N). |
| 7 | Theory § 의 Theorem 1 statement 확정 + proof draft. Paper 1 차 draft (intro + method 완성). |
| 8 | Theorem 2/3 sketch + appendix. Experiments § 작성 + revision + figure 정리. |

---

## 9. Open Questions (resolve 필요)

| # | 질문 | 결정 시점 |
|---|---|---|
| ~~Q1~~ | ~~Target venue~~ — **resolved**: CoRL 2026 | — |
| ~~Q2~~ | ~~Sim 환경~~ — **resolved**: LIBERO (4 suite, Long 1순위) | — |
| Q3 | RLT supervised label 의 LIBERO 적용 — 공개 label 존재 여부 / 없으면 자체 라벨링 필요 | Week 1 |
| ~~Q4~~ | ~~Critical phase ground truth~~ — **resolved**: RLT supervised label (per-step) | — |
| ~~Q5~~ | ~~VLA backbone~~ — **resolved**: π_0 + π_0.5 | — |
| ~~Q6~~ | ~~π_0 / π_0.5 이용 형태~~ — **resolved**: (a) 둘 다 main, 각자 별도 detector + RL refinement | — |
| ~~Q7~~ | ~~LIBERO main suite~~ — **resolved**: LIBERO-Long 1순위, 안정화 후 Spatial/Object/Goal 순차 | — |
| ~~Q8~~ | ~~Theory contribution~~ — **resolved**: Theorem 1 본문 + Theorem 2/3 sketch (§12 detail) | — |

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Self-supervised labeler 가 oracle predicate (또는 RLT label) 과 자주 불일치 | $\varepsilon$ 도출 식 보강 / 다양한 quantile 시도, TLDR encoder 재학습. Theorem 1 의 가정 A1-A3 가 깨지는지 확인. |
| TLDR 약속이 깨짐 (latent ↛ step) | conf metric 으로 OOD 영역 격리, sanity check (Fig 2) 강제. 실패 시 $\varphi$ 재학습 또는 $\lambda = 1$. |
| Few-shot demo 부족 → kernel statistic noisy | $N$ sensitivity ablation 으로 한계 명시, large-$N$ 비교. |
| RLT (직접 경쟁 work) 재현 어려움 | RLT 식만 인용 + 우리 setting 에서 fair comparison 만 (open code 대기 시 footnote). |
| φ collapse (모든 state 가 같은 latent 로 매핑) | $\lambda = 1$ 로 uniformity term 활성화 (§3.1.1). |
| Reviewer "왜 RL 이냐, classifier 면 충분하지 않나" | sanity baseline (단순 frequency 분류기) 와 비교 ablation. |
| π_0 가 LIBERO 에서 이미 success rate 너무 높음 → 실패 trajectory 부족, $B_-$ buffer underflow | LIBERO-Long 을 main 으로 (실패 빈도 ↑), 또는 perturbation/observation noise 로 실패 유도. |
| π_0 vs π_0.5 representation 차이로 detector 가 한쪽에만 작동 | 둘 다 main 으로 결과 보고 (옵션 a), §6.4 #8 cross-VLA 에서 별도 검증. |
| LIBERO obs / action format 차이로 demo loader 재작성 필요 | Week 1 에 LIBERO 공식 dataloader 사용 확정, π_0 / π_0.5 input adapter 만 별도 작성. |
| Theory 가정 (A1 Lipschitz, A2 cover, A3 oracle stability) 이 LIBERO 에서 만족되지 않음 | §12.3 가정의 실험적 검증 — Fig 2 (A1), demo $N$ 변동 (A2), G1 boundary noise 측정 (A3). 위반 시 paper 의 operational guarantee 만 약화 — claim 자체는 유효. |

---

## 11. Out of Scope (명시적 제외)

다음은 본 paper 가 풀지 않음. follow-up paper 로 분리:

- C3 — State-Action joint cell encoder
- G3 — VLA instruction + classifier 라벨러
- $\varphi$ encoder 의 깊은 비교 분석 (TLDR 채택만 정당화, 다른 encoder 는 §6.4 #2 에서 ablation 수준)
- Real-robot 실험 (sim only)
- Multi-modal goal (text + image)
- Tight rate bound (Theorem 1 은 consistency, rate 는 future work)

각 항목은 Future Work reference 로만 등장. detail 은 `reports/alternatives.md`.

---

## 12. Theory Plan (Q8 resolved — Theorem 1 본문 + Theorem 2/3 sketch)

### 12.1 왜 이론을 넣는가

Empirical 결과만으로도 framework 의 동작은 보일 수 있다. 그러나 본 paper 의 핵심 claim 은:
- **"외부 hyperparameter 0"** (§2.2)
- **"few-shot transfer"** (§2.3)

둘 다 데이터에서 자동 도출되는 통계량 ($g$, $\varepsilon$, $h$ 등) 의 *정확성* 에 의존한다. 이 정확성에 수학적 보장이 있어야:
1. Reviewer 가 "왜 이 quantile / bandwidth 가 작동하는가" 를 의심하지 않음 — 이론이 답함.
2. Practitioner 가 새 task 에서 "demo 몇 개 필요한가" 를 추정 가능 — Theorem 1 이 $\rho \asymp \varepsilon$ 조건과 demo 양 관계를 알려줌.
3. Framework 의 적용 한계 (가정 위반 시 어디가 깨지나) 가 명확 — 가정 A1-A3 가 위반되면 어디가 무너지는지 추적 가능.

### 12.2 무엇을 증명할 것인가 — Theorem 1 (본문)

**Theorem 1 (G2 라벨러의 oracle predicate 일치성)**

> **(Informal)**: $\varphi$ 가 step-distance 약속을 충분히 잘 만족하고 (가정 A1), demo 가 goal 영역을 충분히 cover 하며 (가정 A2), oracle predicate 가 영역 boundary 에서 안정적이면 (가정 A3), G2 라벨은 G1 oracle 라벨과 점근적으로 일치한다.

#### 12.2.1 가정의 풀이 (석사생용)

- **(A1) $L$-Lipschitz $\varphi$**: $\|\varphi(s_a) - \varphi(s_b)\| \leq L \cdot d(s_a, s_b)$, 여기서 $d$ 는 step distance.
  - 직관: "$\varphi$ 가 너무 왜곡되지 않았다." $L$ 은 왜곡 정도. 작을수록 좋음.
  - 검증: Fig 2 의 scatter slope.

- **(A2) $\rho$-cover**: demo 마지막 state $\{s_T^{(i)}\}_{i=1}^N$ 이 oracle goal 영역 $\mathcal{G}^* = \{s : f_g(s) = 1\}$ 의 $\rho$-cover.
  - 직관: "demo 가 goal 근처를 골고루 봤다." $\rho$ 가 작을수록 cover 가 dense.
  - 검증: demo $N$ 이 늘면 $\rho$ 가 어떻게 줄어드는지 plot.

- **(A3) Oracle stability**: $f_g$ 는 $\mathcal{G}^*$ 의 $\rho$-neighborhood (in step distance) 에서 constant.
  - 직관: "성공의 정의가 $\rho$ 안에서는 안정적." 즉 boundary 에서 noisy 하지 않음.
  - 검증: G1 predicate 의 boundary 영역 noise 측정.

#### 12.2.2 결론 (formal)

가정 (A1), (A2), (A3) 하, trajectory $\tau$ 의 G2 라벨 $\hat{y}(\tau)$ 와 G1 라벨 $y^*(\tau)$ 에 대해:

$$
\Pr[\hat{y}(\tau) \neq y^*(\tau)] \leq O\left(\frac{L \rho}{\varepsilon}\right) + O\left(\frac{L \varepsilon}{\rho}\right)
$$

#### 12.2.3 결론의 풀이

- 첫 항 $O(L\rho/\varepsilon)$: 임계 $\varepsilon$ 가 너무 좁으면, $\rho$-cover 의 빈틈 때문에 oracle 성공인데 G2 가 실패라 부르는 false negative.
- 둘째 항 $O(L\varepsilon/\rho)$: 임계 $\varepsilon$ 가 너무 넓으면, 목표 영역 밖까지 성공이라 부르는 false positive.
- **$\varepsilon \asymp \rho$ 로 잡으면 두 항 모두 $O(L)$ 로 줄어듦** → 본 paper 의 $\varepsilon$ = demo distance quantile 이 정확히 이 균형을 자동으로 잡는다.
- demo 양이 늘면 $\rho \to 0$ → 일치율 1 로 수렴.

#### 12.2.4 한 줄 요약

> "Demo 가 goal 영역을 충분히 cover 하면, G2 의 라벨 오류는 demo 양이 늘수록 0 으로 수렴한다."

### 12.3 Theorem 2 — Kernel density 일치성 (sketch + appendix)

> **(Informal)**: bandwidth $h_n \to 0$ 적절히 잡으면, kernel-weighted 통계 $\tilde{f}_\pm$ 가 진짜 밀도 $f_\pm^*$ 에 균등 수렴.

이는 표준 kernel density estimation theorem (Silverman 1986, Wand & Jones 1995). 본 paper 의 기여는 새 증명이 아니라 *우리 setting (latent space, buffer 데이터) 에 적용 가능한 형태로 인용*. 본문은 statement 만, 증명은 appendix B.

### 12.4 Theorem 3 — Reward consistency (sketch only)

> **(Informal)**: T1 + T2 → reward $r$ 가 oracle reward $r^*$ 에 수렴. 따라서 RL refinement 의 fixed-point 도 oracle fixed-point 로 수렴.

본문은 corollary 형태로 한 줄. Discussion 에서 "rate 도출은 future work" 명시.

### 12.5 가정의 실험적 검증 (Discussion §)

이론은 가정 셋 (A1, A2, A3) 위에 서 있다. 가정이 LIBERO 에서 실제로 만족되는지를 실험으로 보임:

| 가정 | 검증 방법 | 위치 |
|---|---|---|
| (A1) $\varphi$ Lipschitz | Fig 2 (latent vs step distance scatter) 의 slope = $L$ 추정 | §3.1.4 + Fig 2 |
| (A2) Demo coverage | $N \in \{5, 10, 20\}$ 에서 $\rho$ 변화 plot | §6.4 #6 |
| (A3) Oracle stability | LIBERO success function 의 boundary noise 측정 | Discussion |

### 12.6 Paper 내 위치

- 본문 §4 Theory (0.5 page): Theorem 1 statement + 가정 1 paragraph + Theorem 2/3 의 한 줄 corollary
- Appendix A: Theorem 1 full proof (2-3 pages)
- Appendix B: Theorem 2/3 sketch (1-2 pages)
- Discussion: §12.5 가정의 실험적 검증 (1 paragraph)

### 12.7 작업 일정 (Week 7-8)

- Week 7: Theorem 1 statement 확정 + proof draft
- Week 8: Appendix 정리 + Theorem 2/3 sketch
- Fallback: 시간 부족 시 Theorem 1 만 본문, Theorem 2/3 는 future work 로 축소.

### 12.8 한 줄 — 본 paper 의 이론적 기여

> **"Demo 의 마지막 state 의 평균과 quantile 만으로 정의된 G2 labeler 가, $\varphi$ 가 Lipschitz 이고 demo 가 goal 영역을 cover 하는 한, oracle predicate 와 점근적으로 일치한다."**

이 한 줄이 §2.1 (자동 라벨러) 와 §2.2 (외부 hyperparameter 0) claim 의 수학적 근거.
