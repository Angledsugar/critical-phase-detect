# 대안 옵션 보존 문서

`plan.md` 의 main scope (G2 + C1 + Option (b)) 외 옵션들의 detail. Future Work / ablation 시 참조.

---

## 1. Goal 정의 대안 (G1 · G3 · G4)

### G1 — Outcome Predicate (환경 술어)

**정의**: 환경 디자이너가 명시한 boolean 술어 $f_g : \tau \mapsto \{0, 1\}$.

$$
B_+ = \{\tau : f_g(\tau) = 1\}, \qquad B_- = \{\tau : f_g(\tau) = 0\}
$$

**예시**
- Manipulation: "마지막 state 에서 컵이 손에 잡혀 있다"
- Navigation: "마지막 state 가 목표 영역 내"
- Game / score-based: "최종 score ≥ threshold"

**장점**
- 가장 명확하고 재현 가능; 라벨 노이즈 0
- 환경 사양에서 직접 도출 → 외부 hyperparameter 없음
- 라벨러 신뢰도 상한이 환경 정의의 정확도

**단점**
- task 마다 새 술어 필요 → transfer cost 큼
- 복잡 task (long-horizon manipulation, free-form instruction) 술어 정의 어려움
- VLA 시나리오의 자유 성공 기준과 불일치

**plan 에서의 활용**: **G2 의 oracle baseline ablation**. G2 의 success 라벨이 G1 과 얼마나 일치하는지 비교 → G2 라벨러의 신뢰도 검증.

---

### G3 — VLA Task Instruction + Success Classifier

**정의**: goal 을 자연어 instruction $\ell$, success classifier $h_\theta(\tau, \ell) \in [0, 1]$.

$$
B_+ = \{\tau : h_\theta(\tau, \ell) > 0.5\}, \qquad B_- = \{\tau : h_\theta(\tau, \ell) \leq 0.5\}
$$

**$h_\theta$ 구현 옵션**
- VLA 내장 success token 출력
- 별도 vision-language reward model (RoboCLIP, LIV, VIP)
- Demo contrastive 학습으로 $h_\theta$ 별도 훈련

**장점**
- Multi-task / instruction-following 환경에 자연스러움
- 새 task 전이가 텍스트 변경만으로 가능 (**zero-shot**)
- VLA pipeline 과 직접 통합

**단점**
- $h_\theta$ 자체가 학습 모델 → classifier error 가 framework 전반에 전파
- Calibration 문제: 0.5 임계 자체가 미묘한 hyperparameter
- 학습 부담 추가

**plan 에서의 활용**: **multi-task 확장 실험**. G2 가 single-task 안정화된 후 instruction-following 셋업으로 generalize 검증.

---

### G4 — Implicit (Goal-agnostic black-box labeling)

**정의**: $B_+, B_-$ 가 외부 임의 라벨러 $\mathcal{L}$ 의 출력. Framework 는 internal mechanism 무관:

$$
B_+ = \{\tau : \mathcal{L}(\tau) = \text{success}\}, \qquad B_- = \{\tau : \mathcal{L}(\tau) = \text{failure}\}
$$

**역할**: 단독 옵션이라기보다 **추상화 수준**. G1·G2·G3 의 메타 표현.

**해당 시나리오**
- Software-framework 형태 배포 (사용자가 자기 라벨러 plug-in)
- Heterogeneous 출처 industrial pipeline (human + sim + LLM judge 혼합)
- Theory paper 의 framework 일반성 강조

**plan 에서의 활용**: 논문 method §의 framework 일반성 서술 — "본 detector 는 라벨러-무관" 메시지. Experiments § 에서는 G2 (default), G1 (oracle), G3 (multi-task) 로 instantiate.

---

## 2. State 표현 / Cell 정의 대안

### Option (a) — Discrete cell, φ 는 goal 에만

```
detector input: c_t (정수 ID)
φ 사용처:        (1) B+/B- 라벨링: ‖φ(c_T) - g‖ < ε
               (2) ε 도출
```

**장점**
- 식 변경 0, 학습 부담 최소
- 논문 contribution 이 "buffer-statistic reward" 자체에 집중
- discrete count → kernel 일반화 안 함

**단점**
- 학습 데이터에 없는 새 cell 에서 $\tilde{f} = 0$ → reward 0 → 판정 불가
- φ representation 의 유사도 정보를 detector 가 못 씀

**채택 시기**: VLA 가 아닌 grid/discrete 환경, contribution scope 를 좁히고 싶을 때.

---

### Option (c) — Hybrid (cell_id + φ feature)

```
detector input: (cell_id, φ(c_t))
frequency:     discrete count (Option a 처럼)
```

**장점**: 두 접근 장점 부분 결합
**단점**: 정당화 어색 ("왜 둘 다?"), 일부만 generalize 되어 inconsistency

**평가**: 이론·실용 모두 깔끔하지 않음. 권장하지 않음.

---

### C3 — State-Action joint cell

cell $= (s_t, a_t)$, $\varphi$ 가 state-action 쌍을 encode.

$$
\varphi : S \times A \to \mathbb{R}^d
$$

**장점**
- 가장 expressive — "이 상황에서 이 행동"의 위험성 직접 모델링
- Critical phase 의 정확한 표현 가능 ("위험은 state 만의 함수가 아닌 (s, a) 쌍의 함수")
- VLA 의 action chunk 정보를 직접 활용

**단점**
- 학습 데이터 다량 필요 ($S \times A$ 공간이 큼)
- $\varphi$ 학습 자체가 더 어려움 (action 의 temporal distance 정의가 미묘)
- TLDR / QRL / HILP 모두 state-only 학습이 표준 → 새 loss 설계 필요

**plan 에서의 위치**: **2순위 구현 후보**. G2 + Option (b) 안정화 후 확장.

**구현 방향 hint**
- $\varphi(s_t, a_t)$ 학습 시 positive 쌍은 trajectory 내 가까운 (s, a) 쌍, negative 는 다른 trajectory 쌍
- 또는 state $\varphi_s$ 와 action $\varphi_a$ 분리 후 concat

---

## 3. $\varphi$ encoder 대안

### TLDR (plan 의 default)

Contrastive temporal-distance 학습. $\|\varphi(s_a) - \varphi(s_b)\| \approx$ step distance.

**채택 이유**: 가장 표준적, 식이 simple, 대칭 거리 → kernel 호환 자연스러움.

---

### QRL — Quasi-metric RL

비대칭 거리: $d(s_a, s_b) \neq d(s_b, s_a)$.

**적합 시나리오**
- 단방향 task: 한 번 망가지면 회복 불가 (예: 깨지는 물체)
- Navigation with one-way passages
- Manipulation 의 irreversible action

**단점**
- 식 복잡 (quasimetric 학습 loss)
- 대칭 가정에 의존하는 표준 Gaussian kernel 와 직접 호환 어려움 → asymmetric kernel 필요
- TLDR 보다 구현 복잡

**plan 에서의 활용**: irreversibility 가 critical phase 의 핵심 특성인 task 에서 채택 고려.

---

### HILP — Hilbert Representations

Hilbert space 기반 principled 표현.

**장점**
- 이론 정합성 높음 (inner product, metric 모두 정의)
- 수학적 elegant

**단점**
- Implementation 복잡
- 데이터 효율 TLDR 보다 낮을 수 있음
- Practical 성능 차이가 크지 않음

**plan 에서의 활용**: 이론 강조 논문, Hilbert space 정합 논의 시.

---

### VLA backbone feature 재활용

별도 $\varphi$ 학습 없이 VLA 내부 representation 을 그대로 사용.

**장점**
- 학습 부담 0
- VLA pipeline 과 자연스럽게 통합
- 인퍼런스 시 추가 forward 없음 (이미 backbone 에서 추출)

**단점**
- **치명적**: VLA representation 이 temporal-distance 약속을 만족한다는 보장 없음. 다른 목적 (action prediction) 으로 학습됨
- 약속 미만족 시 plan 의 §3·§4 식이 모두 의미 잃음
- 검증 ablation 필수: VLA feature 의 $\|f(s_a) - f(s_b)\|$ 가 step distance 와 상관관계 있는지

**plan 에서의 활용**: 가장 가벼운 baseline. VLA representation 이 우연히 잘 작동하면 cheap, 안 되면 TLDR 로 후퇴.

---

## 4. 옵션 비교표 (참고용)

이전 plan version 의 G1·G2·G3·G4 비교표 보존:

| 항목 | G1 (Predicate) | G2 (φ-Goal) | G3 (Instruction) | G4 (Implicit) |
|---|---|---|---|---|
| 정의 출처 | 환경 designer | demo 의 $\varphi$-encoding | NL instruction + classifier | 외부 임의 라벨러 (black box) |
| 외부 hyperparameter | 없음 (환경 사양) | 없음 ($\varepsilon$ 이 demo 통계) | classifier 모델 (간접) | 라벨러에 종속 |
| 새 task 전이 | 술어 신규 작성 | demo 만 있으면 자동 | 텍스트 변경 (zero-shot 가능) | 라벨러에 종속 |
| 복잡 task 지원 | 어려움 | 자연스러움 | 자연스러움 | 라벨러에 종속 |
| 라벨러 신뢰성 | 정확 (oracle) | demo quality 의존 | classifier 정확도 의존 | 알 수 없음 (black box) |
| 사전 모델 필요 | 없음 | $\varphi$ encoder | success classifier (또는 VLA) | 라벨러에 종속 |
| 추상화 수준 | 구체 | 구체 | 구체 | 추상 (G1·G2·G3 포함) |
| 단독 실험 가능 | 가능 | 가능 | 가능 | **불가** (구체화 필요) |
| 권장 시나리오 | Sim·단순 task·oracle | VLA + demo 일반 setting | multi-task / instruction-following | framework 추상화·plug-in 라벨러 |

---

## 5. 활용 정책 (plan 본문 외)

| 옵션 | plan 에서의 역할 |
|---|---|
| G2 (main) | 본 plan default, 모든 식의 기준 |
| G1 | Oracle baseline ablation — G2 라벨러 일치율 검증 |
| G3 | Multi-task 확장 실험 |
| G4 | Method § 의 framework 일반성 서술 |
| Option (a) | Discrete 환경 baseline (선택) |
| Option (c) | 비추천 |
| C3 | 2순위 구현 후보 |
| QRL | Irreversible task 에서 고려 |
| HILP | 이론 강조 시 |
| VLA backbone 재활용 | 가벼운 baseline (검증 필수) |
