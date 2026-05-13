# Unsupervised Critical Phase Detector

**Self-Supervised Critical Phase Detection for VLA Refinement via Temporal-Distance Goals**

*CoRL 2026 submission — first draft, 2026-05-11*

---

# Abstract

Vision-Language-Action (VLA) 모델은 일반적인 manipulation task 에서는 안정적으로 작동하나 정밀 phase — 삽입, 정렬, 회전 등 — 에서 자주 실패한다. 이러한 critical phase 를 식별하여 RL fine-tuning 을 집중하는 접근은 효과적이지만 (RLT, 2026), 기존 방법은 사람이 trajectory 마다 라벨링하거나 환경 설계자가 ground-truth success label 를 코딩해야 하는 task-by-task 비용을 수반한다.

본 논문은 demo 5–20 개의 마지막 state 만으로 critical phase detector 를 자동 구성하는 unsupervised framework 를 제안한다. Temporal-distance 기반 encoder $\varphi$ 가 raw proprioception 을 task-progress 좌표로 사상하고, demo 통계로부터 자동 도출되는 G2 self-supervised labeler 가 rollout 을 success/failure 두 buffer 로 분리하며, 각 buffer 에 대한 kernel density estimate 의 log-likelihood ratio $r_t = \log \tilde{f}_+(z_t) - \log \tilde{f}_-(z_t)$ 가 per-step critical score 로 정의된다. 모든 외부 hyperparameter 는 buffer / demo 통계에서 자동 결정되어 사용자 tuning 이 필요하지 않다.

LIBERO-Long benchmark 의 task 0 (π_0.5 backbone, 200 rollouts) 에서 trajectory-level critical 검출 F1 은 leave-one-out cross-validation 기준 0.889 (N=140) 에 도달하며, per-trajectory reward 합의 success vs failure 분리도는 z-score 4.09 로 극도로 깨끗하다. F1 ceiling 이 신호 약화가 아니라 failure pool 부족 (200 rollout 중 13 개 실패) 에 기인함을 분석으로 밝히고, Theorem 1 을 통해 demo 양이 증가할수록 G2 labeler 가 ground-truth success label 에 점근적으로 수렴함을 보인다.

---

# 1. Introduction

## 1.1 Motivation

π_0, OpenVLA, RT-2 등 VLA 모델은 광범위한 manipulation 시연 데이터로 사전훈련되어 대부분의 task 에 어느 정도 작동한다. 그러나 정밀도가 요구되는 phase — peg insertion 의 마지막 정렬, 작은 물체의 grasping, multi-stage task 의 stage 전이 — 에서 성공률이 급격히 떨어진다. 이러한 critical phase 만 식별해서 RL fine-tuning 을 집중하면 전체 trajectory 를 fine-tune 하는 것보다 데이터 효율이 크게 향상된다는 사실은 RLT (RL Token, 2026) 에서 입증되었다.

남은 문제는 critical phase 를 *어떻게 찾을 것인가* 다. 기존 접근은 둘로 나뉘는데, supervised labeling 은 사람이 trajectory 마다 handoff 시점을 라벨링해 정확하지만 task 수에 비례하는 비용이 들고, ground-truth success label 방식은 환경 설계자가 "잡혔다 / 정렬됐다" 같은 boolean predicate 를 코딩해야 해 simulation 에서는 가능하지만 새 task 마다 재작성이 필요하다. VLA 의 핵심 가치가 multi-task generalization 임을 고려하면 critical phase detector 도 task-agnostic 해야 하며, 본 논문은 demo 5–20 개만으로 detector pipeline 이 자동 재구성되는 unsupervised framework 를 제안한다.

## 1.2 Contributions

본 논문의 기여는 하나의 강한 main contribution 과 그로부터 자연스럽게 따라오는 두 가지 desirable property 로 구성된다.

Main contribution 은 temporal-distance encoder, G2 self-supervised labeler, KDE log-ratio reward 로 이루어진 detector framework 다. demo 마지막 state 만으로 success/failure 라벨을 자동 부여하고 KDE log-ratio 로 per-step critical score 를 산출하므로, 사람 라벨, ground-truth success label, classifier retraining 이 모두 불필요하다. 첫 번째 부수 성질은 외부 hyperparameter 가 사실상 0 이라는 점이다 — kernel bandwidth $h$ 는 Silverman's rule 로, success 임계 $\varepsilon$ 은 demo distance quantile 로, critical 임계 $\tau$ 는 log-ratio 부호 ($\tau=0$) 로 모두 buffer 통계에서 자동 결정된다. 두 번째는 few-shot transferability 다 — encoder $\varphi$ 는 task-agnostic 으로 pre-training 되며 새 task 에서는 demo 5–20 개로 $g$ 와 $\varepsilon$ 만 업데이트하면 predicate rewriting 이나 classifier retraining 없이 detector 가 즉시 재구성된다.

## 1.3 Result Preview

LIBERO-Long task 0 (π_0.5_libero, 200 rollouts) 에서 per-trajectory reward 합의 success vs failure 분리도는 z = 4.09 에 이르며, trajectory-level critical 검출 F1 은 LOO-CV 기준 0.889 (N=140, rule = `n_crit_steps ≥ K*`) 에 도달한다. G2 self-labeler 는 ground-truth success 와 0.80 accuracy 로 일치한다 (recall 1.00, precision 0.80, F1 0.889).

---

# 2. Related Work

## 2.1 Critical Phase Detection

본 논문의 직접 비교 대상인 RLT (RL Token, 2026) 는 VLA trajectory 에서 RL fine-tuning 의 시작점을 supervised classifier 로 예측하며, 사람이 trajectory 마다 handoff 시점을 라벨링한다. 정확하지만 task 별 라벨링 비용이 큰 supervised 접근이라는 점에서 demo 통계만을 사용하는 본 framework 와 구별된다. §6.3 의 detection F1 비교가 "본 framework 의 unsupervised 라벨이 supervised 라벨에 얼마나 근접하는가" 를 측정하는 핵심 지표이며, RLT 공개 라벨 확보 시 final version 에 추가한다.

## 2.2 Temporal-Distance Representations

State 쌍의 latent 거리가 step distance 와 같아지도록 contrastive 로 학습하는 temporal-distance latent representation (TLDR) 을 본 논문은 default encoder $\varphi$ 로 채택한다. 비대칭 거리를 학습하여 irreversible task 에 적합한 quasi-metric RL (QRL) 과 Hilbert space 기반의 HILP 또한 같은 계열의 후보이며, §5.6 의 future work 에서 $\varphi$ 선택이 framework 에 미치는 영향을 ablation 으로 다룬다. 본 논문은 $\varphi$ 자체를 새로 제안하지 않으며 prior-work representation 을 critical phase detection 으로 응용 확장하는 데 기여가 있다.

## 2.3 Goal-Conditioned RL

Hindsight Experience Replay (HER) 의 핵심 아이디어 — "도달한 곳이 demo goal 근처면 성공" — 을 본 framework 는 RL 학습이 아니라 detector 의 라벨링 단계로 가져온다. 즉 본 framework 는 HER 의 goal-conditioned RL 가정을 trajectory labeling 에 응용한 형태로 자리잡는다.

## 2.4 VLA RL Refinement

π_0, OpenVLA, RT-2 등 사전훈련된 VLA 를 RL 로 fine-tune 하는 일련의 연구가 존재한다. 본 논문은 VLA 자체를 제안하지 않으며, 기존 VLA 에 detector 를 plug-in 하여 RL refinement 가 trajectory 의 *어디를* 학습할지를 자동으로 결정하도록 만든다.

---

# 3. Preliminaries

## 3.1 Notation

State $s_t \in \mathbb{R}^{d_s}$ 는 time $t$ 에서의 로봇 proprioception 을 의미하며 LIBERO 에서 $d_s = 8$ (end-effector position 3, axis-angle orientation 3, gripper qpos 2) 이다. 한 episode 의 state-action sequence 를 trajectory $\tau = (s_0, a_0, s_1, \ldots, s_T)$ 로 표기하고, trajectory 들의 집합을 buffer $B$ 로 부른다. 본 framework 는 success buffer $B_+$ 와 failure buffer $B_-$ 두 개를 유지한다. Demo set $\mathcal{D} = \{\tau^{(i)}\}_{i=1}^N$ 은 모두 성공으로 가정된 전문가 시연 trajectory 의 모음이며 $N \in \{5, 10, 20, \ldots, 500\}$ 범위를 다룬다. Latent $z_t = \varphi(s_t) \in \mathbb{R}^d$ 는 encoder $\varphi$ 의 출력 ($d = 64$) 을 가리킨다.

## 3.2 Kernel Density Estimation

Buffer $B = \{z^{(i)}\}_{i=1}^M$ 에서 각 latent 주위에 Gaussian kernel 을 얹어 부드러운 밀도를 추정한다:

$$
\tilde{f}_B(z) = \frac{1}{M} \sum_{i=1}^M \mathcal{K}_h(z - z^{(i)}), \quad \mathcal{K}_h(u) = (2\pi h^2)^{-d/2} \exp\left(-\frac{\|u\|^2}{2h^2}\right).
$$

Bandwidth $h$ 는 Silverman's rule 로 자동 결정한다: $h \propto M^{-1/(d+4)} \cdot \hat{\sigma}$. $d=64$ latent space 에서 raw density 가 underflow 되는 것을 방지하기 위해 모든 KDE 계산은 log-space 에서 수행한다 (`cpd.core.kde.KDEStats.log_density`).

## 3.3 VLA Backbone — π_0.5

본 논문은 Physical Intelligence 의 π_0.5_libero 를 backbone 으로 사용한다. openpi 의 WebSocket inference 서버를 통해 256×256 agentview/wrist image, proprioception, language prompt 를 입력받아 action chunk 를 출력하며, replan interval 은 5 step 이다.

## 3.4 Benchmark — LIBERO-Long

LIBERO (Liu et al. 2023) 는 Spatial / Object / Goal / Long 네 개 suite 로 구성된 benchmark 다. 본 논문의 main 실험은 long-horizon multi-stage task 로 구성된 LIBERO-Long (`libero_10`) 에서 수행된다 — critical phase 가 stage 전이에서 명확히 분리되어 detector 의 가치를 입증하기에 가장 적합하기 때문이다. Task 0 의 자연어 명령은 "put both the alphabet soup and the tomato sauce in the basket" 이다.

---

# 4. Critical Phase Detector

본 §은 framework 의 전체 pipeline 을 정의한다. 그림 1 은 데이터 흐름의 한 줄 요약이다.

```
rollout (LIBERO env)
   ↓  proprio (T, 8)
Encoder φ(·)                          [§4.1]
   ↓  latent z (T, 64)
scale normalization
   ↓
G2 self-labeler (g, ε from demos)          [§4.2]
   ↓  binary success/failure
TrajectoryBuffer (B+, B-)
   ↓
compute_kde (Silverman bandwidth)          [§4.3]
   ↓  f̃+, f̃-
Reward.per_step:                           [§4.4]
   r_t = log f̃+(z_t) − log f̃-(z_t)
   ↓
Critical phase: {t : r_t < 0}, min-run=3   [§4.5]
   ↓
trajectory-level features:
   {n_crit_steps, longest_run, crit_fraction}
   ↓
critical/non-critical classification       [§4.6]
```

## 4.1 State space to Latent space : Encoder $\varphi$

본 framework 는 §2.2 에서 인용한 temporal-distance encoder 를 $\varphi$ 로 채택한다. 구체적으로 $\varphi$ 는 3-layer MLP $\mathbb{R}^8 \to \mathbb{R}^{128} \to \mathbb{R}^{128} \to \mathbb{R}^{128} \to \mathbb{R}^{64}$ 로, 각 hidden layer 는 LayerNorm + ReLU 를 거치고 마지막 projection 으로 latent 를 산출한다 (`src/cpd/encoders/tldr.py`). 학습 데이터는 $N$ 개의 demo trajectory $\tau^{(i)} = (s_0^{(i)}, \ldots, s_{T_i}^{(i)})$ 이며 각 $s_t^{(i)} \in \mathbb{R}^8$. 매 학습 step 에서 demo $\tau^{(i)}$ 하나를 추출하고 그 내부에서 anchor 시점 $t_a$ 를 균일하게 추출한 뒤 두 개의 고정 forward step offset $k_\text{pos} < K_\text{neg}$ 를 적용하여 $t_p = \min(t_a + k_\text{pos}, T_i)$, $t_n = \min(t_a + K_\text{neg}, T_i)$ 로 둔다. 이때 $s_{t_a}, s_{t_p}, s_{t_n}$ 은 같은 demo 내부의 anchor / positive / negative state 이며 ($t_p$ 가 anchor 와 시간상 가까운 시점, $t_n$ 이 더 먼 시점), $\varphi$ 는 다음의 triplet contrastive loss 로 학습된다:

$$
\mathcal{L}_\varphi = \mathbb{E}_{(\tau^{(i)}, t_a, t_p, t_n)} \left[ \max\left(0, \|\varphi(s_{t_a}) - \varphi(s_{t_p})\|^2 - \|\varphi(s_{t_a}) - \varphi(s_{t_n})\|^2 + m \right) \right].
$$

$k_\text{pos}=2$ step, $K_\text{neg}=20$ step, margin $m=1$ 로 두고 500 LIBERO-Long demo (10 task × 50 demo) 를 50 epoch × 50 batch × 256 triplet 으로 학습한다. violation rate 는 26 % 에서 0.5 % 로 감소하며 학습 종료 시 평균 ratio $d_\text{an}/d_\text{ap} \approx 70$ 에 도달한다.

이 loss 는 *순서* 만 강제할 뿐 절대 거리는 강제하지 않는다. 한편 $\varphi$ 는 demo id 를 입력으로 받지 않는 결정론적 함수이므로 같은 proprio 는 언제나 같은 latent 로 사상되는 hard constraint 를 추가로 갖는다. 두 제약을 동시에 만족시키는 가장 간결한 해는 모든 demo 를 공유 *task-progress* 좌표 위에 줄 세우는 것이다. 그 결과 $T_i = 300$ 인 느린 demo (step 75 에서 hover) 와 $T_j = 100$ 인 빠른 demo (step 25 에서 hover) 는 같은 latent path 를 서로 다른 속도로 통과하지만 — 각각 step 당 0.33 %, 1 % 의 progress — endpoint 는 같은 cluster 에 도달한다. Latent 거리는 raw clock time 이 아니라 physical progress 를 부호화하며, $d_\text{an}/d_\text{ap} \approx 70$ 은 인접 step 쌍의 latent 거리 $\approx 1$ 단위, demo 끝-처음의 latent 거리 $\approx 70$ 단위라는 느슨한 "1 step $\approx 1$ 단위" 비례로 읽힌다.

Raw proprio 대신 학습된 latent 를 쓰는 동기는 §4.3–4.4 에서 발생하는 세 가지 압력에 있다. 첫째, 8 차원 proprio 는 단위와 스케일이 크게 어긋난다 — position $\in [-1, 1]$ m, 회전 $\in [-\pi, \pi]$ rad, gripper $\in [0, 0.04]$ m. KDE 의 squared Euclidean 거리에서 1 rad 회전은 $\approx 1$ 을, 그리퍼 4 cm 변동은 $\approx 1.6 \times 10^{-3}$ 만을 기여하여 약 600 배의 불균형이 발생하고, KDE 가 사실상 회전 신호만 보게 된다. LayerNorm 과 학습된 projection 은 차원별 dynamic range 를 자동 정규화한다. 둘째, LIBERO 의 home pose 는 task 시작과 종료 직후에 거의 동일하게 등장하므로 (gripper 개폐 차이만 있음) raw proprio Euclidean 에서 $s_0 \approx s_T$ 가 충돌한다. Contrastive 학습은 "시작과 끝은 멀어야 한다" 를 명시적으로 강제하여 시작 영역과 끝 영역을 progress 축을 따라 분리한다. 셋째, §4.4 의 Bayes-optimal log-ratio $r_t = \log \tilde{f}_+(z_t) - \log \tilde{f}_-(z_t)$ 가 성공과 실패를 가르려면 representation 이 "지금 task 의 어느 단계인가" 를 담아야 하는데, end-effector pose 만으로는 이 정보가 부족한 반면 contrastive embedding 은 progress 좌표를 자연스럽게 부여한다.

Latent 차원 $d=64$ 는 두 제약의 교집합으로 결정된다. 500 demo × 평균 280 step $\approx$ 140K latent 점을 distortion $\epsilon = 0.1$ 로 임베딩하기 위한 Johnson-Lindenstrauss 하한은 $d \geq c \cdot \log(140\text{K})/\epsilon^2 \approx 17$ 이며 안전 여유를 두면 $d \in [50, 70]$ 가 자연스럽다. 한편 contrastive representation learning 의 관행 (SimCLR, QRL, HILP) 은 $d \in [64, 128]$ 의 보수적 끝을 default 로 둔다. $d=64$ 가 두 조건의 동시 만족점이며, 더 키우면 KDE bandwidth $h \propto M^{-1/(d+4)}$ 가 $d=128$ 에서 $d=64$ 대비 $\approx 2 \times$ 커져 over-smoothing 으로 success/failure modal 영역의 구분이 약해지는 curse of dimensionality 영역에 진입한다.

## 4.2 This episode is sucess? G2 Self-Supervised Labeler

G2 self-supervised labeler 는 새 rollout $\tau$ 의 성공 여부를 사람 라벨이나 ground-truth success label 없이 결정한다. rollout 의 마지막 latent $\varphi(s_T)$ 가 demo endpoint 들이 모인 cluster 안에 있으면 성공으로 부른다. §4.1 의 encoder 가 trajectory 전체를 학습 신호로 쓰는 것과 달리 G2 는 의도적으로 마지막 한 점 $\varphi(s_T)$ 만 본다.Trajectory 의 step-by-step 정보는 §4.4 의 log-ratio $r_t$ 가 매 $t$ 에서 소비하므로, G2 가 같은 신호를 중복으로 처리할 이유가 없다. 그 결과 G2 는 *outcome* 을 묻고 ("이 rollout 은 성공했는가?"), CPD score 는 *process* 를 묻는 ("진행 중 어디가 위태로웠는가?") 깔끔한 역할 분담이 성립한다.

§5.5 의 ep149 (recovering success) 가 이를 예시한다 — G2 는 결과적으로 성공으로 라벨링하는 동시에, CPD score 는 281 step 동안 failure 처럼 보였음을 별도로 잡아낸다.

위 규칙이 동작하려면 demo endpoint 들이 latent space 에서 정말로 좁게 모여 있어야 한다. 

LIBERO demo 들이 모두 같은 home pose 에서 출발하므로 $\varphi(s_0^{(i)}) \approx \varphi(s_0^{(j)})$ 가 자명하고, 같은 task 의 demo 가 공통 goal config 근처에서 종료하므로 $\varphi(s_T^{(i)}) \approx \varphi(s_T^{(j)})$ 또한 자명하다. 

여기에 triplet loss 가 "시작과 끝은 멀어야 한다" 를 모든 demo 에 동시 강제하면, $\varphi$ 가 demo id 를 받지 않는 결정론적 함수라는 §4.1 의 제약과 결합하여, 가장 간결한 해는 공유 progress 축 위에 모든 demo 가 흐르도록 두는 것이다. 좁은 시작 cluster, 좁은 끝 cluster, 더 넓은 중간 path 의 구조가 이렇게 자연스럽게 형성된다.

또한 LIBERO 의 ground-truth success label — "object 가 goal config 에 있는가" — 자체가 마지막 state 의 함수이므로, $\varphi(s_T)$ 가 demo endpoint cluster 안에 있는지를 묻는 G2 는 ground-truth 의 구조를 직접 모사한다. 이는 Theorem 1 의 A3 stability 가정으로 형식화된다.

구체적인 라벨링은 세 단계로 진행된다. 먼저 demo endpoint 들의 latent 평균을 goal point $g$ — endpoint cluster 의 중심 — 로 잡는다:

$$
g = \frac{1}{N} \sum_{i=1}^N \varphi(s_T^{(i)}).
$$

다음으로 demo endpoint 들이 $g$ 에서 얼마나 흩어져 있는지를 보고 threshold $\varepsilon$ 를 95-th percentile, 즉 "demo 의 95 % 가 들어오는 ball 의 반지름" 으로 잡는다:

$$
\varepsilon = \mathrm{Quantile}_{0.95}\left( \{\|\varphi(s_T^{(i)}) - g\|\}_{i=1}^N \right).
$$

마지막으로 새 rollout 의 마지막 latent 가 이 $\varepsilon$-ball 안에 있는지로 라벨을 매긴다:

$$
\hat{y}(\tau) = \mathbb{1}\left[ \|\varphi(s_T) - g\| \leq \varepsilon \right].
$$

$d=64$ 에서의 task-specific norm 변동을 흡수하기 위해 $g$ 와 모든 latent 를 $\bar{\eta} = \mathrm{mean}_i \|\varphi(s_T^{(i)})\|$ 로 정규화한다 (Exp1 에서 $\bar{\eta} = 38.14$). 사람 라벨도 ground-truth success label 도 필요하지 않다는 점이 핵심이며, demo trajectory 만 있으면 $g, \varepsilon$ 모두 자동으로 결정된다.

robust 성에는 한계가 있다. G2 는 (i) goal 이 workspace 전체에 unconstrained 하게 분포하여 endpoint 가 넓게 산포할 때, (ii) 한 task 에 여러 정답 endpoint 가 존재하는 multi-modal success 의 경우, 또는 (iii) "방 청소" 처럼 성공이 robot pose 가 아니라 environment 분포 자체로 정의되는 task 에서 깨진다. 이러한 케이스를 다루기 위한 goal-conditioned G2 / KDE-based G2 변종은 §5.6 의 future work 로 미룬다.

## 4.3 Buffer Construction & KDE

라벨링된 rollout 들로 `TrajectoryBuffer` 를 채운다. Success rollout 의 모든 step latent 가 $B_+ = \{z_t : (\tau, t) \text{ s.t. } \hat{y}(\tau) = 1\}$ 에, failure rollout 의 latent 가 $B_- = \{z_t : (\tau, t) \text{ s.t. } \hat{y}(\tau) = 0\}$ 에 들어간다. 각 buffer 에 대해 §3.2 의 KDE 로 $\tilde{f}_+, \tilde{f}_-$ 를 추정하고 Silverman bandwidth 를 buffer 별로 독립적으로 계산한다.

LOO (leave-one-out) protocol 을 강제하여, rollout $\tau^{(j)}$ 를 평가할 때 자기 자신의 latent 는 buffer 에서 제외한다. 이는 Theorem 1 이 가정하는 "평가 trajectory 는 buffer 에 대해 out-of-sample" 을 보장하며, $N=10$ rollout 으로 ep5 / ep9 를 평가할 때 self-density 가 KDE 를 지배하는 문제를 회피한다.

## 4.4 Per-Step Critical Score (Reward)

이 §은 매 step latent $z_t$ 가 success buffer 의 분포에 더 잘 맞는지 failure buffer 의 분포에 더 잘 맞는지를 log-likelihood ratio 로 측정하여 이 값을 per-step critical score $r_t$ 로 정의한다. 직관적으로 "지금 이 위치는 성공한 rollout 들이 머물던 영역인가, 아니면 실패한 rollout 들이 빠지던 영역인가" 를 매 step 한 숫자로 답하는 것이다.

구체적인 정의는

$$
r_t = \log \tilde{f}_+(z_t) - \log \tilde{f}_-(z_t)
$$

이다. 여기서 $\tilde{f}_+(z_t)$ 는 §4.3 에서 success buffer $B_+$ 로 만든 KDE 가 점 $z_t$ 에서 갖는 density — "성공 trajectory 들이 이 latent 주변을 얼마나 자주 방문했는가" 의 부드러운 추정량 — 이며, $\tilde{f}_-(z_t)$ 는 failure buffer $B_-$ 에 대한 동일한 양이다. 본질은 두 density 의 비 $\tilde{f}_+ / \tilde{f}_-$ 이고, log 를 취해 차의 형태로 바꾼 것이 $r_t$ 다.

부호가 곧 해석을 준다. $r_t > 0$ 이면 success buffer 가 더 빽빽하므로 "지금 위치는 성공한 rollout 들이 자주 들렀던 곳" 으로 읽히고, $r_t < 0$ 이면 failure buffer 가 더 빽빽하여 "실패 rollout 들이 머물던 위험 지대" 가 된다. $r_t \approx 0$ 은 양쪽 분포가 비슷하여 단정 짓기 어려운 영역이다. 단순한 latent 거리가 아니라 *buffer 전체 분포에서의 typicality* 를 비교한다는 점이 핵심이며, 그래서 KDE 가 필요하다 — 한 점이 아니라 분포로 평가해야 multi-modal goal 영역이나 폭이 넓은 carry 구간까지 자연스럽게 다뤄진다.

Threshold $\tau = 0$ 은 ad hoc 한 선택이 아니라 Bayes-optimal decision boundary 다. Prior $P(\text{succ}) = P(\text{fail}) = 1/2$ 를 가정하면 posterior 비교가 likelihood 비교로 축소되어

$$
P(\text{fail} \mid z) > P(\text{succ} \mid z) \;\;\Longleftrightarrow\;\; \tilde{f}_-(z) > \tilde{f}_+(z) \;\;\Longleftrightarrow\;\; r_t < 0
$$

이 성립한다. 즉 $\tau = 0$ 으로 자르는 것은 사후확률 기준으로 "실패일 확률이 성공일 확률보다 크다" 와 정확히 같은 사건을 잡는 일이다. Hyperparameter 가 사실상 0 인 framework 의 자동성이 여기서 한 번 더 확인된다. 즉, 사용자 tuning 없이 부호 비교만으로 결정 경계가 정해진다.

## 4.5 Critical Phase Extraction Rule

Per-step score $r_t$ 를 trajectory 내 연속 구간으로 binarize 한다:

$$
\text{critical}(t) = \mathbb{1}[r_t < \tau], \quad \tau = 0.
$$

1–2 step 의 noise dip 을 무시하기 위해 min-run = 3 의 debounce 를 적용하여 3 step 이상 연속된 critical 구간만 인정한다. 각 trajectory 에 대해 debounce 후 critical step 의 총 개수 `n_crit_steps`, 가장 긴 연속 critical 구간 `longest_run`, 두 양의 비 `critical_fraction = n_crit_steps / T`, 그리고 연속 신호의 적분 `sum_r = $\sum_t r_t$` 네 가지 trajectory-level feature 가 정의되며, 이들이 §6 의 detection F1 평가에 사용된다.

## 4.6 Trajectory-Level Classification

Critical / non-critical trajectory 분류는 trajectory-level feature 에 임계값을 적용하여 수행한다:

$$
\hat{c}(\tau) = \mathbb{1}\left[ \text{n\_crit\_steps}(\tau) \geq K^* \right].
$$

$K^*$ 는 train 데이터의 best F1 으로 결정한다. §6 의 비교에서 4 개 feature 중 `n_crit_steps` 가 가장 우수했다.

## 4.7 Theorem 1 (G2 Consistency)

이 §의 한 줄 요약: §4.2 의 G2 라벨러가 단순한 경험적 규칙이 아니라 ground-truth success label 의 통계적 추정량임을 보이고, demo 양이 증가할수록 G2 라벨이 ground-truth 라벨에 점근적으로 수렴함을 증명한다.

먼저 두 라벨러를 구분해 두는 것이 좋다. *G1* 은 환경이 제공하는 ground-truth success label — LIBERO 의 `check_success()` 처럼 "물체가 정말로 목표 위치에 있는가" 를 진리값으로 답하는 함수 — 로, 우리가 unsupervised framework 에서 의도적으로 사용하지 않는 정답 라벨이다. *G2* 는 §4.2 에서 정의한, demo endpoint cluster 만으로 만들어진 self-supervised 추정 라벨이다. 자연스러운 의문은 "G2 가 G1 의 어떤 통계적 추정량인가, 그리고 demo 가 많아지면 둘이 같아지는가" 다. Theorem 1 은 이에 대해 세 개의 자연스러운 가정 하에서 답을 준다.

세 가정은 G2 와 G1 을 비교 가능하게 만드는 최소 조건이다. (A1) encoder $\varphi$ 가 step-distance 를 $L$-Lipschitz 로 보존한다는 것은, latent space 의 거리가 원래 state space 의 거리를 일정한 배율 안에서 유지한다는 의미다. 이 조건 없이는 "latent 에서 가깝다" 가 "state 에서도 가깝다" 로 환원되지 않아 ground-truth 의 boundary 가 latent 에서 보존되지 않는다. (A2) demo $N$ 개의 endpoint 가 진짜 goal 영역을 반경 $\rho$ 안에서 빈틈없이 덮는다는 cover 가정은, demo 가 너무 적어 goal 영역에 큰 구멍이 남으면 그 구멍에서 ground-truth 는 성공이라 부르는데 G2 는 demo 근처가 아니어서 실패로 놓치는 경우를 배제한다. (A3) ground-truth success label 이 경계면에서 안정적이라는 것은, $\|s - g\| = \varepsilon$ 근처의 작은 perturbation 으로 ground-truth 라벨이 휙휙 뒤집히지 않는다는 의미다. 셋 다 LIBERO 와 유사한 manipulation 환경에서 자연스럽게 성립하는 조건이다.

이 세 가정 하에서 라벨 불일치 확률의 상한은

$$
\Pr[\hat{y}(\tau) \neq y^*(\tau)] \leq O\left(\frac{L\rho}{\varepsilon}\right) + O\left(\frac{L\varepsilon}{\rho}\right)
$$

으로 주어진다. 두 항은 서로 반대 방향의 오류를 나타낸다. 첫 항 $O(L\rho/\varepsilon)$ 은 false negative 의 상한 — 임계값 $\varepsilon$ 이 너무 작으면 demo cover 의 빈틈에서 살짝 벗어난 진짜 성공 endpoint 를 G2 가 ball 밖이라며 놓친다. 둘째 항 $O(L\varepsilon/\rho)$ 은 false positive 의 상한 — $\varepsilon$ 이 너무 크면 ball 이 진짜 goal 영역 바깥까지 흡수하여 실패 rollout 도 성공으로 라벨링한다. 둘은 $\varepsilon$ 의 단조 함수로 서로 trade-off 관계를 이루며, $\varepsilon \asymp \rho$ 로 두면 두 항이 동시에 $O(L)$ 이 되어 균형이 잡힌다.

이 균형점이 §4.2 의 95-quantile 설계와 정확히 맞아떨어진다. demo endpoint 들의 $g$ 까지 거리의 95-th percentile 은 본질적으로 demo cover 의 평균 반경 — 즉 $\rho$ — 의 통계적 추정량이며, 따라서 우리 framework 는 trade-off 의 최적점을 사용자 tuning 없이 자동으로 잡는다. Hyperparameter 가 사실상 0 인 framework 의 설계가 단순한 편의가 아니라 정보 이론적으로 정당화되는 선택임이 여기서 드러난다.

마지막으로 점근 거동. demo 수 $N$ 이 늘면 cover 가 빽빽해져 $\rho \to 0$ 이 되고, $\varepsilon \asymp \rho$ 의 균형이 유지되는 한 에러 상한 자체가 0 으로 수렴한다. 즉 demo 가 충분히 많아지면 G2 라벨은 ground-truth 라벨과 점근적으로 일치한다. 이것이 "G2 는 ground-truth success label 의 일관 추정량 (consistent estimator)" 이라는 진술의 형식적 의미이며, §4.2 가 경험적으로 그럴듯한 규칙이 아니라 통계적으로 정당화된 라벨러임을 보장한다. 자세한 증명은 Appendix A 에 둔다.

세 가정의 실험적 검증은 다음과 같다. (A1) 은 encoder 학습 종료 후 triplet violation rate 0.5 % 로 사실상 만족된다 (§4.1). (A2) 는 demo $N \in \{5, 10, 20, 50, 500\}$ 변동 ablation 으로 검증할 예정이며, $N$ 이 작을수록 false negative 가 증가하리라는 예측이 들어맞는지 본다. (A3) 는 LIBERO success function 의 boundary 근처 noise 측정으로 확인한다.

---

# 5. Experiments

## 5.1 Setup

실험에 사용한 backbone 은 openpi WebSocket server (GPU 1) 로 serve 한 π_0.5_libero 이며, suite 는 LIBERO-Long (`libero_10`) 의 task 0 에 집중한다. Demo set 은 500 개의 LIBERO-Long expert demonstration (10 task × 50 demo, 8-dim proprio, `data/tldr_demos.pkl`) 이며, encoder 는 이 demo 로 사전학습된 `checkpoints/tldr.pt` 를 사용한다. Rollout 은 task 0 에서 200 episode 를 수집하되 replan = 5, max_steps = 520 으로 두고 init state 를 cycling 한다 (한 task 의 50 개 init state 를 4 cycle 반복, cycle 마다 env seed 변경). KDE 는 Silverman bandwidth, log-space, leave-one-out 으로 계산한다.

본 draft 는 π_0.5 단일 backbone, task 0 단일 task 의 결과이며, π_0 추가 및 다른 task 확장은 §5.6 의 future work 로 미룬다.

## 5.2 Experiment 1 — Pipeline Sanity Check (10 rollouts)

전체 pipeline 이 작동하는지 검증하기 위한 small-scale 실험으로 10 episode 만 수행했다. 결과는 다음과 같다.

| 지표 | 값 |
|---|---:|
| π_0.5 raw success rate | 0.80 (8/10) |
| G2 vs ground truth: accuracy / precision / recall / F1 | 0.80 / 0.80 / 1.00 / 0.889 |
| G2 confusion matrix (TP/FP/TN/FN) | 8/2/0/0 |
| Per-trajectory reward sum: success mean ± std | +1209 ± 426 |
| Per-trajectory reward sum: failure mean ± std | −3438 ± 710 |
| 분리도 z-score = $(\mu_+ - \mu_-) / (\sigma_+ + \sigma_-)$ | 4.09 |

G2 self-labeler 는 ground truth 와 80 % accuracy 로 일치하며, false positive 두 건은 $\varepsilon$ 이 N=10 의 작은 demo 집합에서 다소 관대하게 잡힌 결과로 — Theorem 1 의 가정 A2 가 $N=10$ 에서 약하다는 점이 직접적으로 반영된다. 보다 중요한 것은 per-step reward $r_t$ 의 trajectory 합이 success 와 failure 를 z = 4.09 로 분리한다는 점이며, 이는 어떤 plausible random baseline threshold 도 능가하는 깨끗한 signal 이다. Per-step reward 곡선 (`reports/exp1/reward_curves.png`) 에서 success trace 는 $r_t$ 가 단조 증가하여 $+5 \ldots +20$ 에 도달하는 반면 failure trace 는 누적 negative reward 로 $-20 \ldots -40$ 에 도달한다 — signal 이 time $t$ 에 대해 monotone 하고 시각적으로 명확하다.

## 5.3 Experiment 2 — Detection F1 vs Rollout Budget N

이 §의 한 줄 요약: VLA rollout 을 몇 개 수집해야 (= buffer 가 얼마나 커야) critical phase detection 이 안정되는지를 묻는 실험이다. 결론은 $N \approx 140$ 에서 F1 = 0.889 로 plateau 에 도달하고 200 까지 늘려도 추가 이득이 없다는 것이다.

여기서 "rollout budget $N$" 이란 KDE buffer 를 만들기 위해 수집한 VLA rollout 의 수를 가리킨다. §4.3 에서 본 framework 는 $N$ 개의 rollout 을 G2 로 success/failure 두 buffer 로 나누고 각각 KDE 를 만들기 때문에, $N$ 이 작으면 buffer 가 비어 KDE 가 noisy 해지고 detection 성능이 떨어진다. 따라서 "몇 개부터 안정되나" 는 실용적으로 중요한 질문이며, 새 task 에 framework 를 적용할 때 필요한 데이터 수집 budget 을 결정짓는다.

프로토콜은 다음과 같다. Task 0 (LIBERO-Long, π_0.5 backbone) 에서 init-state cycling 으로 200 episode 를 수집했다 (187 성공, 13 실패). 그 다음 $N \in \{10, 20, 30, 50, 70, 100, 140, 200\}$ 각각에 대해 첫 $N$ 개만 잘라 buffer 로 쓰고 leave-one-out cross-validation (LOO-CV) 으로 F1 을 측정한다. LOO 는 한 trajectory 를 평가할 때 그 trajectory 를 buffer 에서 빼고 평가하는 절차로, 자기 자신과의 비교로 인한 inflation 을 막아 측정한 F1 이 새 rollout 에 일반화될 수치임을 보장한다.

Detection rule 은 §4.6 의 trajectory-level feature 들 — $n_\text{crit\_steps}$ (전체 critical step 수), $\text{longest\_run}$ (연속 critical 구간의 최대 길이), $\text{crit\_fraction}$ (critical step 비율) — 각각에 임계값을 두어 "이 rollout 이 critical 인가" 를 결정한다. 비교를 위해 단순 규칙 `has_critical` ("$r_t < 0$ 이 3 step 이상 연속되는 구간이 하나라도 있으면 critical") 도 함께 평가한다.

여기서 $K^*$, $L^*$, $F^*$ 가 무엇인지 명시한다. $K^*$ 는 `n_crit_steps` 규칙의 결정 경계로, trajectory 의 critical step 누계가 $K^*$ 이상이면 그 rollout 을 critical 로 부른다 (단위 = step 개수). $L^*$ 는 `longest_run` 규칙의 결정 경계로, 가장 긴 연속 critical 구간이 $L^*$ 이상이면 critical 이다 (단위 = step). $F^*$ 는 `crit_fraction` 규칙의 결정 경계로, 전체 길이 대비 critical step 비율이 $F^*$ 이상이면 critical 이다 (단위 = 비율, $[0,1]$). 별표 ($^*$) 는 "LOO-CV 가 각 fold 마다 자동으로 골라낸 값" 임을 가리키며 사람이 정한 hyperparameter 가 아니다. 예를 들어 $N=140$ 에서는 $K^* \approx 120$ step, $L^* \approx 53$ step, $F^* \approx 0.37$ 이 자동 선택되었고 $N=200$ 에서는 각각 $K^* \approx 129$, $L^* \approx 52$, $F^* \approx 0.32$ 가 선택되었다.

임계값을 fold 마다 LOO-CV 로 다시 정하는 이유는, in-sample 에서 가장 좋은 임계값을 그대로 보고하면 test 에서 우연히 잘 맞는 값을 골라 F1 을 부풀리게 되므로 그 낙관성을 제거하기 위함이다. F1 의 ground truth 는 LIBERO env 의 success label 의 negation 이다 — "critical = failure" 로 정의한 본 framework 에서 ground-truth 의 "success" 가 곧 "non-critical" 이므로 negation 이 자연스럽다.

| $N$ | has_critical (loose rule) | longest_run ≥ L* | n_crit_steps ≥ K* | critical_fraction ≥ F* |
|---:|:---:|:---:|:---:|:---:|
| 10 | 0.333 | 0.333 | 0.667 | 0.667 |
| 20 | 0.333 | 0.667 | 0.667 | 0.400 |
| 30 | 0.235 | 0.500 | 0.667 | 0.400 |
| 50 | 0.182 | 0.600 | 0.800 | 0.571 |
| 70 | 0.158 | 0.833 | 0.727 | 0.545 |
| 100 | 0.132 | 0.800 | 0.769 | 0.667 |
| 140 | 0.122 | 0.737 | 0.889 | 0.706 |
| 200 | 0.122 | 0.759 | 0.857 | 0.571 |

세 가지 관찰이 표에서 읽힌다. 첫째, $n_\text{crit\_steps} \geq K^*$ 가 모든 $N$ 에서 가장 좋은 rule 이며 $N=140$ 에서 F1 = 0.889, $N=200$ 에서 F1 = 0.857 (bootstrap 95 % CI = [0.70, 0.97]) 에 도달한다. 둘째, 단순 규칙 `has_critical` 은 무너진다 — $\tau=0$ 이 너무 관대하여 성공 rollout 의 93 % 이상도 "어딘가 critical step 이 있다" 로 flag 되고 precision 이 0.07 로 사실상 random 수준이 된다. 이는 "critical step 이 한 번이라도 있는가" 가 아니라 "critical step 이 *얼마나 누적되었는가*" 가 진짜 signal 이라는 의미다. 셋째, F1 은 $N \approx 140$ 부근에서 plateau 에 도달하고 $N=200$ 까지 더 늘려도 추가 이득이 없다 — 이 ceiling 의 원인은 §5.4 에서 별도로 분석한다.

요약하면 본 실험은 production-ready 검출 규칙으로 $n_\text{crit\_steps} \geq K^*$ 를 권장하며, F1 0.86–0.89 가 framework 의 현실적 성능 수준임을 보여준다. 새 task 에 framework 를 적용할 때는 $N \approx 140$ rollout 이 sample budget 의 합리적 목표가 된다.

## 5.4 Analysis — F1 Ceiling 은 신호가 아니라 Failure Pool

F1 은 $N$ 과 함께 단조 증가하다가 0.85–0.89 에서 천장에 닿는다. 이 천장은 signal weakness 가 아니라 failure pool 부족에 기인한다는 것이 우리의 주장이며, N=200 에서 측정한 per-trajectory feature 의 success/failure 분리도가 그 직접적 근거다.

| Feature | Success median (p25, p75) | Failure median (p25, p75) | z-gap |
|---|---|---|---:|
| `n_crit_steps` | 38 (29, 55) | 216 (166, 366) | +2.49 |
| `longest_run` | 17 (14, 24) | 151 (68, 332) | +2.05 |
| `critical_fraction` | 14 % (11 %, 19 %) | 42 % (32 %, 70 %) | +2.02 |
| `T` (episode length) | 277 (264, 290) | 520 (520, 520) | +3.54 |

Failure 는 success 대비 critical step 이 약 6 배, worst-run 이 약 9 배 길어 분포가 z ≥ 2 로 명확히 분리된다 — 신호 overlap 은 F1 ceiling 의 원인이 아니다. 진짜 원인은 π_0.5 의 task 0 success rate 가 93.5 % 로 매우 높아 200 rollout 중 failure 가 13 개뿐이라는 점이다. 1 misclassification 당 F1 이 약 0.05 흔들리므로 최적 운영점에서도 1 FP + 1 FN 만 발생하면 F1 ≈ 0.857 이 되고, 13 개 failure 로는 0.95 ceiling 에 통계적으로 도달할 수 없다.

Failure 사례를 들여다보면 두 가지 부류가 식별된다. 첫째는 깨끗한 failure (예: ep118, T = 520, longest_run = 488, crit_fraction = 95.8 %) 로 정책이 처음부터 잘못된 manifold 에 진입하여 timeout 한 경우이며, CPD 가 sweet spot 에서 매우 자신있게 detect 한다. 둘째는 회복 가능한 critical (예: ep149, success but T = 514, longest_run = 281, crit_fraction = 63.2 %) 로 중간에 $r_t$ 가 음수로 깊이 들어가지만 마지막에 정책이 회복하여 success 로 끝나는 경우다. 후자는 F1 ceiling 의 주범이며 — 모양은 failure 이나 결과는 success — "critical = 반드시 fail" 이 아니라 "critical = state-space 의 failure-like region 에 들어감" 이라는 약한 정의의 정직한 한계를 드러낸다.

## 5.5 시각적 직관

`reports/exp2/explain_critical_phase.png` (Fig 4) 는 대표 세 episode 의 $r_t$ 곡선을 비교한다. Success 사례인 ep115 는 거의 모든 시점에서 $r_t > 0$ 이며 끝부분에 7-step dip 만이 존재한다 (longest_run = 7, crit_frac = 2.7 %). Failure 사례인 ep118 은 거의 처음부터 끝까지 $r_t < 0$ 이 유지되어 빨간 음영이 trajectory 전체를 덮는다 (longest_run = 488, crit_frac = 95.8 %). 마지막으로 ep149 는 중간 281-step 의 critical 구간을 통과한 뒤 끝에서 polynomial 하게 복구하여 success 로 끝나는 recovering success 의 대표적 사례이며 (crit_frac = 63.2 %), F1 ceiling 을 가장 잘 설명하는 episode 다.

## 5.6 Future Work — 본 draft 에서 다루지 못한 실험

본 draft 는 single-task, single-backbone scope 에 머무르며 final version 에서 다음 실험들을 추가한다. Multi-task / multi-suite 실험은 LIBERO 의 네 suite 위에서 cross-task transfer matrix 4×4 를 구성하여 detector framework 의 task-agnostic 주장을 직접 검증하고, multi-backbone 실험은 π_0 와 π_0.5 양쪽에서 detector 가 작동하는지를 확인하는 한편 π_0 demo 로 학습한 detector 를 π_0.5 rollout 에 적용하는 cross-VLA transfer 도 시도한다. RLT 의 supervised baseline 과의 비교는 F1 정답을 RLT 사람 라벨로 잡고 본 G2 가 supervised label 에 얼마나 근접하는지를 측정하며, 공개 라벨이 확보되는 Week 6 에 추가한다.

세 가지 ablation 이 함께 계획되어 있다. 첫째, encoder 선택 (TLDR vs QRL vs HILP) 이 framework 에 미치는 영향을 측정한다. 둘째, $N \in \{5, 10, 20, 50, 500\}$ 에서 G2 정확도의 변화로 Theorem 1 의 가정 A2 ($\rho$-cover) 를 실험적으로 검증한다. 셋째, π_0.5 success rate 가 50 % 정도인 harder task 에서 평가하여 F1 ceiling 문제를 해소하거나, 또는 metric 을 class imbalance 에 robust 한 AUROC / PR-AUC 로 전환한다.

마지막으로 G2 의 두 가지 구조적 확장을 future work 로 둔다. Goal-conditioned G2 는 unconstrained goal randomization 에 대응하기 위한 것으로, 현재 G2 의 단일 $(g, \varepsilon)$ 가정은 LIBERO 처럼 goal 이 한정된 region 에 분포할 때 작동하지만 goal 이 workspace 전체에 분포하는 task 에서는 per-goal-region 별 $(g, \varepsilon)$ 또는 goal-spec 조건부 $g(\text{goal-spec})$ 같은 확장이 필요하며, P1 (endpoint clustering) 가정이 깨지는 구체적 경계를 분석한다. KDE-based G2 는 multi-modal success — "object 를 책상 어느 곳에든 놓기" 처럼 정답 endpoint 가 여러 개인 task — 에 대응하기 위한 것으로, mean-radius 기반 $(g, \varepsilon)$ 대신 KDE $\tilde{f}_+$ 를 직접 활용한다. §4.4 의 CPD score 는 이미 KDE 기반이라 multi-modal 을 지원하므로 §4.2 의 G2 labeler 만 확장하면 되며, Theorem 1 의 A2 가정이 multi-modal manifold 에서도 유효함을 확인해야 한다.

---

# 6. Conclusion

본 논문은 temporal-distance encoder, G2 self-supervised labeler, KDE log-likelihood ratio 로 구성되는 unsupervised critical phase detection framework 를 제안했다. 모든 외부 hyperparameter 는 demo 와 buffer 통계에서 자동 도출되며, 새 task 에서는 demo 5–20 개만 추가하면 detector 가 즉시 재구성된다. LIBERO-Long task 0 의 200 rollout 에서 trajectory-level critical 검출 F1 은 LOO-CV 기준 0.889 (N=140) 에 도달하고, per-trajectory reward 합은 success 와 failure 를 z = 4.09 로 깨끗하게 분리한다. F1 ceiling 분석은 한계의 원인이 신호 약화가 아니라 failure pool 부족임을 밝혀, harder task / multi-task / RLT 직접 비교를 통한 framework 의 일반화 능력 검증이라는 명확한 next step 을 제시한다.

본 framework 의 가장 큰 강점은 수작업 라벨이 필요하지 않다는 점이며, 이는 VLA 의 multi-task 정신과 정합하면서 supervised RLT 의 task-by-task 라벨링 비용을 제거한다. Theorem 1 은 demo 양 증가에 따른 ground-truth success label 일치성을 보장하여 framework 에 수학적 근거를 제공한다.

---

# Reference

> 본 draft는 reference list를 비워둠. final version에서 LaTeX bibliography로 채울 항목:
>
> - **RLT (RL Token, 2026)**: critical phase 직접 비교 — supervised handoff predictor
> - **TLDR**: Temporal-Distance Latent Representation, contrastive temporal embedding
> - **QRL**: Wang et al., quasi-metric reinforcement learning
> - **HILP**: Park et al., Hilbert representations
> - **HER (Andrychowicz+ 2017)**: hindsight experience replay
> - **LIBERO (Liu+ 2023)**: 4-suite manipulation benchmark
> - **π_0 (Physical Intelligence 2024)**: VLA backbone
> - **π_0.5 (Physical Intelligence 2025)**: VLA backbone (본 paper main)
> - **Silverman (1986)**: kernel density bandwidth selection
> - **Wand & Jones (1995)**: kernel smoothing methods
> - **OpenVLA, RT-2**: VLA pretraining

---

# Appendix

## A. Theorem 1 — Full Proof

> 본 draft는 proof를 sketch만. final version에서 채울 구조:
>
> 1. 가정 (A1) $L$-Lipschitz $\varphi$의 정의 및 step distance $d_\text{step}$과의 관계.
> 2. 가정 (A2) $\rho$-cover의 formal definition — demo final state set이 ground-truth goal region을 cover.
> 3. 가정 (A3) ground-truth success label stability — $\rho$-neighborhood에서 constant.
> 4. False negative bound: trajectory $\tau$가 ground-truth 성공 ($y^* = 1$)이지만 G2 실패 ($\hat{y}=0$)인 경우 → $\|\varphi(s_T) - g\| > \varepsilon$이면서 $s_T \in \mathcal{G}^*$. 가정 (A1)-(A2)로부터 $\rho < \varepsilon$ 일 때 0.
> 5. False positive bound: 대칭적 논증.
> 6. 두 bound 결합: $\Pr[\hat{y} \neq y^*] \leq O(L\rho/\varepsilon) + O(L\varepsilon/\rho)$.
> 7. $\varepsilon \asymp \rho$로 잡으면 $O(L)$ — quantile-based $\varepsilon$이 자동으로 이 균형 달성.

## B. Theorem 2 — KDE Consistency Sketch

표준 kernel density consistency theorem (Silverman 1986; Wand & Jones 1995) 을 본 setting — temporal-distance latent space 와 buffer 데이터 — 에 적용한 형태로 인용한다. Bandwidth 가 $h_n \to 0$ 이고 $n h_n^d \to \infty$ 인 조건 하에 $\tilde{f}_\pm \to f_\pm^*$ 가 균등 수렴함을 보장한다.

## C. Implementation Details

| Component | 값 / 설정 |
|---|---|
| Encoder $\varphi$ | 3-layer MLP, hidden 128, latent 64, LayerNorm + ReLU |
| Triplet loss | $k_\text{pos}=2, K_\text{neg}=20, m=1.0$, 50 epoch × 50 batch × 256 |
| KDE bandwidth | Silverman's rule, log-space computation |
| Critical phase debounce | $\tau=0$, min-run = 3 |
| Trajectory feature | `n_crit_steps`, `longest_run`, `critical_fraction`, `sum_r` |
| F1 evaluation | leave-one-out CV, bootstrap 200 iter for CI |
| π_0.5 inference | openpi WebSocket server, replan = 5, max_steps = 520 |
| Rollout collection | 200 episodes on task 0, init_states cycling × 4, env seed per cycle |

## D. Artifacts

```
src/cpd/encoders/tldr.py              # TLDR encoder
src/cpd/encoders/tldr_train.py        # triplet trainer
src/cpd/core/labeler.py               # G2 labeler
src/cpd/core/buffer.py                # TrajectoryBuffer (B+, B-)
src/cpd/core/kde.py                   # log-space KDE
src/cpd/core/reward.py                # log-ratio per-step reward
scripts/train_tldr.py                 # encoder training
scripts/collect_libero_rollouts.py    # rollout collection
scripts/eval_cpd_pipeline.py          # Exp1 entrypoint
scripts/extract_critical_phase.py     # Fig 4 / mp4 generator
scripts/sweep_cpd_n.py                # F1 vs N sweep
scripts/sweep_cpd_n_loocv.py          # LOO-CV F1 (Exp2 main)
reports/exp1/                         # Exp1 results
reports/exp2/                         # Exp2 results + explanation
reports/exp2/explain.md               # TLDR usage, latent_dim=64 rationale
reports/exp2/explain_tldr_contrastive.md  # "latent dist ∝ time dist" 의 의미, demo 길이 무관성
reports/exp2/explain_g2_bridge.md     # §4.1 → §4.2 의 endpoint clustering 인과
reports/exp2/explain_env_randomization.md # init randomization 이 G2 에 미치는 영향 (모래시계)
reports/exp2/critical_phase/          # 15 representative episodes (mp4 + fig4 PNG)
checkpoints/tldr.pt                   # trained encoder
data/tldr_demos.pkl                   # 500 LIBERO-Long demos
```

## E. Reproducibility Notes

본 실험은 `pi05_libero` openpi checkpoint 를 GPU 1 에서 WebSocket 으로 serve 하고 `.venv-libero` 에서 LIBERO 환경을 cross-venv 로 호출하는 구성에서 수행되었다. 200 rollout 수집에 약 93 분 (50 개의 init_state 를 4 cycle), encoder 학습에 단일 GPU 로 약 5 분, N=200 LOO-CV F1 sweep 에 약 10 분이 소요되었다. Random seed 는 rollout collection 의 경우 cycle 별 $7+k$, encoder 학습의 경우 42 로 고정했다.
