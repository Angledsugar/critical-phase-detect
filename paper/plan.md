# Critical Phase Detector — RL 설계안 (G2 메인)

Trajectory 생성기에서 나오는 궤적을 보고 **현재 state가 위험 상태인지** 판정하는 detector를 강화학습으로 학습한다. Goal과 success 라벨링은 demo 기반 temporal-distance embedding(G2)으로 자동 도출하며, 모든 hyperparameter는 buffer·demo 통계에서 결정된다.

> **Scope**: G2 (φ-distance goal) + C1 (cell = state) + Option (b) (state 가 φ-embedded vector)
> **다른 옵션** (G1·G3·G4·C3·discrete cell 등): §10 Future Work, 별도 보존 문서 `reports/alternatives.md` 참조

---

## 1. 문제 정의

본질적으로는 trajectory-level outcome → per-state label 의 **credit assignment 문제**이며, action이 환경에 영향을 주지 않으므로 엄밀한 MDP는 아니다. RL framework로 감싸는 이유는 (a) state 일반화 (φ 공간에서의 보간), (b) downstream control 정책의 safety shield 연동을 위한 표준 인터페이스 확보.

본 plan은 **G2 (Goal State via Temporal-Distance Embedding)** 를 default goal 정의로, **state $= \varphi(s_t) \in \mathbb{R}^d$** (TLDR encoder 출력)을 default state 표현으로 채택한다. 모든 식의 hyperparameter 는 buffer·demo 통계에서 도출되어 외부 도입 항이 없다.

---

## 2. RL 구성요소

### 2.1 Warmup buffer

- **Positive buffer** $B_+ = \{\tau_1^+, \dots, \tau_m^+\}$: G2 라벨러로 성공 분류된 trajectory.
- **Negative buffer** $B_- = \{\tau_1^-, \dots, \tau_n^-\}$: 실패 분류된 trajectory.
- 각 trajectory는 state 시퀀스 $\tau = (s_1, s_2, \dots, s_{|\tau|})$, 여기서 $s_t$ 는 visual + proprioception 관측.
- **Demo set** $D = \{\tau^d_1, \dots, \tau^d_N\}$: VLA expert 또는 human teleop 으로 얻은 expert trajectory. 본 plan은 **few-shot 가정 ($N \in [5, 20]$)** 을 기본으로 한다.

### 2.2 Goal 정의 — G2 (Temporal-Distance Embedding)

Demo 마지막 state 들의 latent embedding 평균을 anchor 로 사용:

$$
g = \frac{1}{N}\sum_{i=1}^N \varphi\bigl(s_T(\tau^d_i)\bigr)
$$

trajectory $\tau$의 success 판정:

$$
\text{success}(\tau) = \mathbb{1}\bigl[\,\|\varphi(s_T(\tau)) - g\| < \varepsilon\,\bigr]
$$

$\varepsilon$의 demo-derived 도출:

$$
\varepsilon = \mathrm{Quantile}_q\Bigl(\bigl\{\|\varphi(s_T(\tau^d_i)) - g\|\bigr\}_{i=1}^N\Bigr), \quad q = 1 - \frac{1}{N}
$$

분위 $q$ 자체가 demo 표본 크기에서 결정되므로(single outlier 만 경계 외) 외부 hyperparameter 가 아님.

### 2.3 State / Action

- **State**: $\varphi(s_t) \in \mathbb{R}^d$ — TLDR encoder 의 latent embedding (§3 참조).
- **Action**:
  - `0`: 평범한 state (안전)
  - `1`: 위험한 state

원 plan 의 "current cell" 추상화를 φ-embedded continuous space 로 generalize. OOD state 는 §3.3 신뢰도 평가로 검출.

---

## 3. Temporal-Distance Encoder $\varphi$

### 3.1 $\varphi$ pre-training (TLDR-style loss)

**목표**: $\varphi : S \to \mathbb{R}^d$ 가 다음 약속을 만족:

$$
\|\varphi(s_a) - \varphi(s_b)\| \;\approx\; \bigl(s_a \to s_b\,\text{의 step distance}\bigr)
$$

**학습 데이터**: $\mathcal{B} = B_+ \cup B_- \cup D$ (warmup 후 모든 trajectory).

#### 3.1.1 Loss 구성

**(i) Alignment — within-trajectory distance regression**

같은 trajectory 의 두 state $(s_t, s_{t'})$ 의 latent 거리가 step 거리와 일치하도록 회귀:

$$
\boxed{\;
\mathcal{L}_\text{align}(\varphi) \;=\; \mathbb{E}_{\tau \in \mathcal{B}} \;\mathbb{E}_{(t, t') \sim [1, |\tau|]^2}\Bigl[\bigl(\|\varphi(s_t) - \varphi(s_{t'})\| - |t - t'|\bigr)^2\Bigr]
\;}
$$

샘플 $(t, t')$ 를 trajectory 전체 범위에서 균등 샘플링하면 모든 시간 거리 $0 \leq |t-t'| \leq |\tau|-1$ 가 학습 신호로 들어옴 → 단일 loss 만으로 full distance scale 학습.

**(ii) Uniformity — cross-trajectory margin (collapse 방지, optional)**

Alignment 만으로는 모든 trajectory 가 latent 의 한 영역으로 collapse 할 수 있음 (특히 trajectory 들이 비슷한 start state 공유 시). 다른 trajectory 의 state 쌍이 적어도 margin $M$ 만큼 떨어지도록 보강:

$$
\mathcal{L}_\text{uniform}(\varphi) \;=\; \mathbb{E}_{\substack{\tau \neq \tau'\\ s \in \tau,\, s' \in \tau'}}\Bigl[\bigl(\max(0,\; M - \|\varphi(s) - \varphi(s')\|)\bigr)^2\Bigr]
$$

**Margin $M$ 의 buffer-derived 도출** (외부 hyperparameter 회피):

$$
M = \mathrm{Mean}_{\tau \in \mathcal{B}}\,|\tau|
$$

직관: "다른 trajectory state 들은 평균적으로 trajectory 한 개 길이만큼 떨어져 있어야 한다" 는 prior.

**(iii) 총 loss**

$$
\boxed{\;\mathcal{L}_\varphi \;=\; \mathcal{L}_\text{align} + \lambda \cdot \mathcal{L}_\text{uniform}, \qquad \lambda \in \{0, 1\}\;}
$$

- **Default**: $\lambda = 0$ (alignment 만). MVP 단계에서 collapse 가 관측되지 않으면 충분.
- **Collapse 관측 시**: $\lambda = 1$ 로 uniform term 활성화. 추가 sweep 불필요.

#### 3.1.2 샘플링 protocol

Mini-batch 마다 (per step):

1. $\mathcal{B}$ 에서 trajectory $K$ 개 샘플 ($K = 32 \sim 64$ 권장)
2. 각 trajectory 에서 random 위치 쌍 $(t, t')$ 균등 샘플 → alignment positive
3. ($\lambda = 1$ 인 경우) 다른 trajectory 와 cross 쌍 균등 샘플 → uniformity negative
4. Positive : negative 비율 1:1

#### 3.1.3 Architecture 가이드라인

- **Visual obs**: ResNet-18 또는 ViT-Small backbone (ImageNet pretrained 권장 → 데이터 효율)
- **Proprio + low-dim**: 3-layer MLP, hidden dim 256, ReLU
- **Output dim** $d \in \{64, 128\}$ (TLDR 원논문 권장 범위)
- **Final layer**: linear, no activation (latent 거리가 자유롭게 scale up)
- **LayerNorm**: 마지막 hidden 에 적용 (학습 안정성)

#### 3.1.4 학습 후 sanity check

학습된 $\varphi$ 가 약속을 만족하는지 검증:

| 검증 항목 | 기대 값 | 측정 |
|---|---|---|
| Demo 인접 step 거리 | $\approx 1$ | $\mathrm{Median}_D\{\|\varphi(s_t) - \varphi(s_{t+1})\|\}$ |
| Demo 시작-끝 거리 | $\approx \|\tau^d\| - 1$ | demo 별 비교 |
| Cross-trajectory 평균 거리 | $\geq M$ ($\lambda = 1$ 시) | random pair 통계 |
| 같은 task 다른 trajectory 의 진행도 비슷 시점 | 작은 거리 | $\|\varphi(s_t^{\tau_1}) - \varphi(s_t^{\tau_2})\|$ |
| Step distance vs latent distance correlation | Pearson $r > 0.9$ | scatter plot 검증 |

위 1·2·5 가 만족되면 §3.2 의 $h = \mathrm{Median}\{\|\varphi(s_t) - \varphi(s_{t+1})\|\}_D$ 가 자연스럽게 1 근방, §3.3 의 conf metric 이 의미 있음. 만족 안 되면 (i) data 부족, (ii) architecture mismatch, (iii) $\lambda = 1$ 로 전환 중 점검.

### 3.2 Kernel bandwidth $h$ 도출

§4.2 의 kernel-weighted 통계용 Gaussian kernel:

$$
K(z_a, z_b) = \exp\!\left(-\frac{\|z_a - z_b\|^2}{2 h^2}\right)
$$

bandwidth $h$ 의 demo-derived 도출 — demo 내부 인접 step 사이 거리의 중앙값:

$$
h = \mathrm{Median}\bigl\{\|\varphi(s_t) - \varphi(s_{t+1})\| : (s_t, s_{t+1})\,\text{인접 in } D\bigr\}
$$

직관: TLDR 약속이 성립하면 $\|\varphi(s_t) - \varphi(s_{t+1})\| \approx 1$ (1 step 거리). 따라서 $h$ 는 φ 공간에서 "1 step 단위" 의 길이 — hyperparameter 가 아니라 demo 통계로부터 도출된 자연 단위.

### 3.3 OOD 영역 신뢰도

φ 약속이 깨지는 영역 (demo·buffer 에서 멀리 떨어진 state, 다른 task / 환경):

$$
\mathrm{conf}(s) = \exp\!\left(-\frac{\min_{s' \in D \cup B_+ \cup B_-} \|\varphi(s) - \varphi(s')\|}{h}\right) \in (0, 1]
$$

$\mathrm{conf}(s) < \tau_{\mathrm{conf}}$ ($\tau_{\mathrm{conf}}$ 은 demo 자체의 conf 분포 5-percentile 로 도출) 인 state 에서는 detector 출력 신뢰도 낮음 → downstream 에서 fallback ("모르겠으면 위험" 처리 권장).

---

## 4. Buffer 통계량 (kernel-weighted)

### 4.1 Trajectory 가중치 (길이 반영)

$$
L^+_{\min} = \min_{\tau \in B_+} |\tau|, \qquad L^-_{\min} = \min_{\tau \in B_-} |\tau|
$$

$$
w_+(\tau) = \frac{L^+_{\min}}{|\tau|}, \qquad w_-(\tau) = \frac{L^-_{\min}}{|\tau|} \quad \in (0, 1]
$$

직관:
- 짧은 성공 = 깨끗한 안전 경로 → 가중치 1.
- 긴 성공 = 위험 우회 → 감쇠.
- 짧은 실패 = 위험 즉시 진입, state 당 책임 큼 → 1.
- 긴 실패 = 헤매다 실패, state 당 책임 희석 → 감쇠.

> 극단적 outlier 가 신경 쓰이면 $L^\pm_{\min}$ 대신 buffer 의 하위 분위(예: 10%) 사용. 분위도 buffer 로 결정되므로 외부 hyperparameter 아님.

### 4.2 Kernel-weighted 방문 빈도

discrete cell counting 을 φ 공간 kernel-weighted density 로 generalize:

$$
\tilde{f}_+(s) = \frac{\displaystyle\sum_{\tau \in B_+} w_+(\tau) \cdot \frac{1}{|\tau|}\sum_{s' \in \tau} K\bigl(\varphi(s), \varphi(s')\bigr)}{\displaystyle\sum_{\tau \in B_+} w_+(\tau)}
$$

(대칭으로 $\tilde{f}_-(s)$ 정의.)

직관:
- "$s$와 비슷한 state 들이 (가중 평균으로) 얼마나 자주 등장하는가"
- $K$ 가 indicator 함수면 원 plan 의 discrete frequency 로 환원 → 자연스러운 generalization
- OOD state 도 부분적으로 보간 ($K$ 를 통해 가까운 학습 state 로부터)

---

## 5. Reward 설계

### 5.1 Reward A — per-step (dense)

$$
\boxed{\;r_t^{(A)} = (2 a_t - 1)\,\bigl(\tilde{f}_-(s_t) - \tilde{f}_+(s_t)\bigr)\;} \in [-1, 1]
$$

- 짧은 실패에 자주 등장하는 state ($\tilde{f}_-$ 큼): $a_t = 1$ 이 강한 +.
- 짧은 성공에 자주 등장하는 state ($\tilde{f}_+$ 큼): $a_t = 0$ 이 강한 +.
- 양 buffer 비슷하거나 OOD: 신호 ≈ 0 (식별 불가능 state 에 적절).

### 5.2 Reward B — trajectory-level (sparse, episode 끝)

"위험으로 찍힌 state 를 빼면, 잔여 길이가 demo trajectory 길이와 비슷해야 한다" 가설.

$$
S(\tau, a) = |\tau| - \sum_{t=1}^{|\tau|} a_t
$$

$$
\boxed{\;
R(\tau, a) =
\begin{cases}
-\,w_+(\tau)\cdot \dfrac{\sum_t a_t}{|\tau|} & \tau \in B_+ \\[10pt]
-\,w_-(\tau)\cdot \dfrac{\displaystyle\min_{\tau^d \in D}\bigl|\,S(\tau, a) - |\tau^d|\,\bigr|}{|\tau|} & \tau \in B_-
\end{cases}
\;}
$$

- $\tau \in B_+$: 1 을 찍을수록 감점. 짧은 성공일수록 압력 강.
- $\tau \in B_-$: 위험 제거 후 잔여 길이가 *어떤* demo 길이와 일치하면 0. 짧은 실패일수록 정확도 압력 강.
- 분모 $|\tau|$ 로 길이 정규화.
- Reference 로 $D$ (demo) 사용 — expert 길이는 안정적인 reference. ($B_+$ 는 G2 라벨러 noise 반영하므로 noisy.)

### 5.3 통합 reward

$$
\boxed{\;
r_t = \underbrace{(2 a_t - 1)\bigl(\tilde{f}_-(s_t) - \tilde{f}_+(s_t)\bigr)}_{\text{A: 어디가 위험인가}} + \underbrace{\mathbb{1}[t = |\tau|]\cdot R(\tau, a)}_{\text{B: 얼마나 많이 위험인가}}
\;}
$$

A·B 는 상보적이다. A 만 쓰면 위험 라벨 개수 편향, B 만 쓰면 어느 state 인지 짚지 못함.

---

## 6. Edge case: Positive-only ($|B_-| = 0$)

$\tilde{f}_- \equiv 0$, $w_-$ / $L^-_{\min}$ 미정의 → 통합 reward 가 "모두 안전" 으로 degenerate. 이는 **데이터의 본질적 한계**이며 reward 결함이 아니다 (위험은 contrast 로만 식별 가능).

운영 분기:

```
if |B_-| == 0:
    detector 학습 보류. 필요시:
    r_t = (2 a_t - 1)(mean_s f_+(s) - f_+(s_t))로 한정 운영 (alarm 용도)
elif |B_-| 작음:
    옵션 (b1): demo perturbation 으로 합성 negative 부트스트랩
    옵션 (b2): trajectory generator stochasticity 증가로 실패 능동 수집
else:
    위 통합 reward 로 본격 학습
```

(G2 라벨러 도입으로 $\varepsilon$ 이 너무 관대해 모두 $B_+$ 분류되는 경우도 동일 분기. §3.3 OOD 신뢰도와 결합해 진단.)

---

## 7. Sanity check

| 케이스 | 기대 동작 | 식 확인 |
|---|---|---|
| $B_+$ 에만 등장하는 state | 안전으로 학습 | $\tilde{f}_-(s)\to 0,\ \tilde{f}_+(s)>0 \Rightarrow a_t=0$ 이 + |
| $B_-$ 에만 등장하는 state | 위험으로 학습 | $\tilde{f}_+(s)\to 0,\ \tilde{f}_-(s)>0 \Rightarrow a_t=1$ 이 + |
| 양쪽에 동일 빈도/가중 | 학습 안 함 | $r^{(A)} = 0$ |
| 미관측 / OOD state | 신뢰도 낮음, fallback | $\mathrm{conf}(s) < \tau_{\mathrm{conf}}$ |
| 모든 성공 길이 동일 | 길이 가중 자동 비활성 | $w_+ \equiv 1$, 일반 frequency 로 환원 |
| 매우 긴 실패 outlier | 가중 감쇠로 영향 약 | $w_- = L^-_{\min}/|\tau| \to 0$ |
| State $= \varphi(s_t)$ continuous | $K$ kernel 통한 보간 | discrete frequency 로 환원 가능 (indicator $K$) |
| Demo 일치 trajectory | 성공 라벨 | $\|\varphi(s_T) - g\| \approx 0 < \varepsilon$ |
| φ 약속 깨지는 OOD | conf 기반 fallback 필수 | conf metric 신뢰도 평가 |

---

## 8. 학습 파이프라인 요약

1. **Warmup**: trajectory 생성기 rollout 으로 raw $B_+, B_-$ 채움 (sparse outcome 으로 분류). Demo 셋 $D$ 준비.
2. **$\varphi$ pre-training**: TLDR contrastive loss 로 $\varphi$ 학습 ($B_+ \cup B_- \cup D$).
3. **G2 라벨링**:
   - $g = \frac{1}{N}\sum_i \varphi(s_T(\tau^d_i))$ 계산
   - $\varepsilon$ = quantile derivation
   - $B_+ / B_-$ 재분류 (raw 라벨과 일치율 검증)
4. **통계 사전계산** (1회): $L^\pm_{\min}$, $w_\pm(\tau)$, kernel bandwidth $h$, conf threshold $\tau_{\mathrm{conf}}$, $\tilde{f}_\pm(s)$.
5. **Detector 학습**: PPO/DQN 등 표준 알고리즘. State $= \varphi(s_t)$, Action $\in \{0, 1\}$, Reward $= r_t^{(A)} + \mathbb{1}[t=|\tau|]\cdot R$.
6. **Buffer 갱신** (옵션): 새 rollout 들어오면 $\varphi$ 재학습 또는 freeze 후 통계만 update.

---

## 9. 구현 시 주의

- **Sanity baseline**: RL 학습 전, $\tilde{f}_-(s) > \tilde{f}_+(s)$ 단순 분류기의 성능 측정 → RL 이 못 이기면 RL 도입 정당성 재검토.
- **Class imbalance**: 위험 state 희소 → false negative cost > false positive. 평가에서 비대칭 metric (recall-우선).
- **OOD state**: §3.3 conf 기반 fallback 필수 ("모르겠으면 위험").
- **Kernel bandwidth**: demo-derived $h$ 가 default. Demo 너무 적을 때 Silverman's rule 보조 가능.
- **$\varphi$ 신선도**: trajectory generator 갱신 시 $\varphi$ 재학습 권장. Frozen 모드는 distribution shift 검증 후.
- **G2 라벨러 신뢰도 평가**: $\varepsilon$ 이 너무 관대하면 false-success 다발 → sparse outcome (G1 oracle) 가능한 환경에서 일치율 측정 ablation.

---

## 10. Future Work

본 plan 에서는 다루지 않지만 후속 연구로 확장 가능한 방향. detail 은 `reports/alternatives.md` 참조:

- **C3 — State-Action joint cell**: cell $= (s_t, a_t)$, $\varphi$ 가 state-action 쌍을 encode. 더 powerful 하지만 학습 데이터 다량 필요. 본 framework 의 cell 정의만 교체하면 가능. **2순위 구현 후보**.
- **G1 — Outcome Predicate**: 환경 술어 기반 oracle 라벨러. G2 baseline ablation 으로 활용.
- **G3 — VLA Instruction Classifier**: 자연어 instruction 으로 zero-shot 라벨링. Multi-task 환경에 유용.
- **G4 — Implicit framework abstraction**: 라벨러 black box, plug-in interface. 논문 method §의 일반성 서술용.
- **다른 $\varphi$ encoder**: QRL (asymmetric distance), HILP (Hilbert space), VLA backbone feature 재활용.
- **Discrete state representation (Option a)**: cell discrete + $\varphi$ 는 goal 에만. 식 단순하지만 OOD 일반화 약함.
