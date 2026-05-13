# Exp2 — 설명 문서 (Critical phase / TLDR / latent_dim 선택)

석사생 수준에서도 이해할 수 있도록 풀어 쓴 보충 자료. paper §3.3 (φ encoder) / §3.5 (CPD rule) 본문을 작성할 때 그대로 가져다 쓸 수 있는 형태.

목차:
1. [CPD가 "critical phase"라고 판단하는 기준](#1-cpd가-critical-phase라고-판단하는-기준)
2. [TLDR encoder는 어디에 쓰이는가](#2-tldr-encoder는-어디에-쓰이는가)
3. [왜 proprio 8차원을 latent 64차원으로 늘리는가](#3-왜-proprio-8차원을-latent-64차원으로-늘리는가)

---

## 1. CPD가 "critical phase"라고 판단하는 기준

### 한 줄로

> "지금 이 state가, 이전에 **실패했던 trajectory들의 state**와 더 닮았으면 critical."

### 사전 지식 (개념 4개)

1. **TLDR encoder φ(·)**: state (proprio 8-dim) → latent z (64-dim). 비슷한 task progress를 가진 state들이 latent space에서 가까이 모이도록 학습된 encoder. (§2에서 더 자세히)

2. **Buffer**: 지금까지 모은 rollout들을 **success bucket (B+)** 과 **failure bucket (B-)** 두 통에 GT label로 나눠 담아둠.

3. **KDE (Kernel Density Estimation) f̃+, f̃−**: 각 bucket의 latent들로부터 "이 latent z 근처에 success state가 얼마나 빽빽한가 / failure state가 얼마나 빽빽한가"를 부드러운 확률 밀도로 만든 것. Silverman bandwidth로 자동 결정.

4. **Per-step CPD score**:

   ```
   r_t = log f̃+(z_t) − log f̃−(z_t)
   ```

   - `r_t > 0`  →  이 latent 근처에 success state가 더 많음 → "괜찮은 state"
   - `r_t < 0`  →  failure state가 더 많음 → "수상한 state"
   - `r_t = 0`  →  반반 (decision boundary)

### 작동 원리 (3 step)

```
  state s_t  →  φ(s_t) = z_t  →  r_t = log f̃+(z_t) − log f̃−(z_t)
                                  ↓
                            r_t < 0 인가?
                                  ↓ (yes)
                       3 step 이상 연속이면 critical
```

- **임계값 τ=0**: log-ratio 부호로 갈라치기. 직관은 "success-쪽 밀도가 failure-쪽 밀도보다 작아진 순간".
- **min-run=3**: 한두 step만 삐끗하는 것은 노이즈로 보고 무시. 3 step 이상 지속돼야 진짜 위험 구간으로 인정.

### 그림으로 직관 잡기

`reports/exp2/explain_critical_phase.png` 참고. task00 200ep 중 대표 3개:

**[위] ep115 — 깨끗한 success (T=264, longest critical run=7, crit fraction=2.7%)**
- 거의 모든 시점에서 r_t > 0 (success 분포에 잘 머무름)
- 끝부분에만 잠깐 dip이 있지만 7 step짜리라 metric 측면에서도 약함
- 직관: 정책이 demo의 success manifold를 따라 잘 굴러감

**[가운데] ep118 — 깨끗한 failure (T=520, longest=488, crit fraction=95.8%)**
- 거의 처음부터 끝까지 r_t < 0 (failure 분포에 깊이 빠져 있음)
- 빨간 음영이 전체를 덮음
- 직관: 정책이 처음부터 잘못된 manifold로 진입해서 빠져나오지 못함 (timeout으로 실패)
- 이런 episode가 CPD의 sweet spot — 매우 자신있게 critical로 분류

**[아래] ep149 — 까다로운 success (T=514, longest=281, crit fraction=63.2%)**
- 중간부터 r_t가 음수로 내려가서 한참 거기 머무름 (critical 구간이 281 step)
- 그러나 마지막에 r_t가 폭발적으로 양수로 올라옴 — 정책이 어떻게든 회복해서 성공
- **F1 ceiling의 주범**: 모양은 failure처럼 생겼지만 결과는 success → false positive
- "critical phase에 빠져도 회복 가능한 경로가 있다"는 정직한 현실 반영

### 왜 이 정의가 합리적인가

**Theorem 1과의 연결 (paper plan §3.4)**

log-ratio는 정확히 Bayes-optimal classifier에서 나오는 score:

```
P(failure | z) > P(success | z)
⟺ log f̃−(z) > log f̃+(z)
⟺ r_t < 0
```

prior가 같다고 가정하면 τ=0이 자연스러운 default.

**KDE 자체의 의미**

"이 latent 근처에 비슷한 경험이 얼마나 많았나"를 부드럽게 측정. 단순히 nearest-neighbor가 success/failure 보고 결정하는 것보다 안정적 — bandwidth가 알아서 noise를 평균냄.

**min-run=3의 의미**

KDE는 부드럽지만 latent trajectory에는 fluctuation이 있음. min-run을 두지 않으면 1-step dip이 잔뜩 잡혀서 precision이 박살남 (실제로 `has_critical` 룰이 그렇게 망가졌고 F1=0.12). min-run=3은 "수상한 구간이 의미있게 지속되었는가"를 보는 안전장치.

### 한계 (paper에 솔직히 써야 할 것)

- **failure → recovery가 가능한 trajectory** (위 ep149)는 본질적으로 CPD가 false positive 낼 수밖에 없음. "critical = 반드시 fail"이 아니라 "critical = state-space의 failure-like region에 들어감"이라는 약한 정의.
- **τ=0은 prior balance 가정**. 우리 buffer는 success:fail = 187:13으로 극도로 imbalanced — 엄밀히는 log-prior offset (`log(N+/N-)`)을 빼줘야 함. paper writing 단계에서 정리할 포인트.
- **KDE bandwidth는 dimension에 민감** — 64-dim에서 Silverman이 정말 적절한지는 별도 ablation 필요 (§3 참고).

---

## 2. TLDR encoder는 어디에 쓰이는가

TLDR은 **CPD 파이프라인의 첫 단계인 latent encoder φ(·)**. raw state → latent z. 그 뒤 모든 단계(KDE, reward, critical phase 판정)는 latent z 위에서 돌아감.

### 코드 위치

| 역할 | 파일 |
|---|---|
| 모델 정의 (φ 자체) | `src/cpd/encoders/tldr.py` — `TLDREncoder` (MLP) |
| 학습 loop (triplet contrastive loss) | `src/cpd/encoders/tldr_train.py` — `TLDRTrainer` |
| 학습 entrypoint | `scripts/train_tldr.py` (Hydra config `configs/encoder/tldr.yaml`) |
| 학습 데이터 | `data/tldr_demos.pkl` (500 LIBERO-Long expert demos, proprio (T,8)) |
| 가중치 | `checkpoints/tldr.pt` |
| 데이터 추출 스크립트 | `scripts/extract_libero_proprio.py` (hdf5 → pickle) |
| 추상화 base | `src/cpd/encoders/base.py` — ablation으로 QRL / HILP로 교체 가능 |

### 런타임에 실제로 호출되는 곳 (rollout state → latent)

1. `scripts/eval_cpd_pipeline.py` — main 평가 파이프라인
2. `scripts/extract_critical_phase.py` — fig4 / mp4 generator
3. `scripts/sweep_cpd_n.py` — Exp2 (F1-vs-N) 스크립트
4. `/tmp/explain_cp.py` — §1의 critical phase 곡선 그릴 때 (one-off)

### 파이프라인에서의 위치

```
rollout (LIBERO env)
   ↓  proprio: (T, 8)  ← robot eef_pos(3) + axis-angle(3) + gripper(2)
TLDREncoder.encode(·)                  ← 여기가 TLDR 등장 지점
   ↓  z: (T, 64)
scale normalization (z / mean‖z_T‖)    ← demo 종점 norm으로 나눠줌
   ↓
TrajectoryBuffer.add(z, success)       ← B+ / B- bucket
   ↓
compute_kde(buffer) → f̃+, f̃−          ← Silverman bandwidth
   ↓
Reward.per_step(z_t) = log f̃+(z_t) − log f̃−(z_t)
   ↓
threshold τ=0 + min-run=3 → critical phase intervals
```

### 왜 raw state로 KDE 안 하고 굳이 TLDR을 거치나

세 가지 이유. 각각이 독립적으로 raw proprio를 부적합하게 만듦.

#### 이유 1: 단위가 너무 달라서 KDE의 "거리"가 깨짐

LIBERO proprio 8차원의 구성과 스케일:

| 차원 | 의미 | 단위 | 값 범위 |
|---|---|---|---|
| `ee_pos` (3) | end-effector 위치 | m | [-0.5, +0.5] |
| `ee_ori` (3) | axis-angle 회전 | rad | [-π, +π] |
| `gripper_qpos` (2) | 그리퍼 폭 | m | [0, 0.04] |

KDE는 두 점 사이 Euclidean distance를 봄: `dist² = Σ(xᵢ−yᵢ)²`. 차원별 스케일이 100배씩 차이나니까 같은 1단위 변화도 distance 기여가 천차만별:

| 사건 | squared diff 기여 |
|---|---:|
| 그리퍼 fully open→close (0.04 변화) | 0.0016 |
| 로봇이 1cm 움직임 (0.01 변화) | 0.0001 |
| 로봇이 1rad 회전 | 1.0 |

→ KDE는 사실상 **회전 성분만 보고 그리퍼는 무시함**. Pick-and-place task에서 그리퍼 상태가 결정적인데 못 봄.

이거만 fix하려면 차원별 z-score 정규화로 충분. 하지만 이유 2, 3 때문에 그걸로는 부족.

#### 이유 2: raw proprio는 "지금 자세"만, TLDR latent는 "task의 어느 단계"를 인코딩 ← 핵심

**문제 시나리오**: pick-and-place task. 로봇이 home에서 시작 → 물건 잡으러 감 → 옮김 → home으로 복귀.

**raw proprio에서 보면**:
- 시작 state와 끝 state의 raw 값이 거의 동일함 (둘 다 home position + gripper open)
- KDE: "이 raw state는 success 분포에서 자주 나옴" — 시작인지 끝인지 구분 못함
- 실패 trajectory가 중간에 home 근처에서 헤맸다면, success/failure cloud가 home 근처에서 둘 다 빽빽 → log-ratio ≈ 0 → CPD가 정보 없음

**TLDR latent에서 보면**:
- TLDR은 triplet loss로 "시간상 가까운 state쌍 → latent 거리도 가깝게"를 학습
- 그 결과 latent space는 task progress의 1차원 곡선처럼 펴짐 (effective intrinsic dim ≈ 1~5, ambient 64에 임베딩)
- 시작 state → latent의 한쪽 끝, 끝 state → latent의 다른 쪽 끝. raw에서 똑같던 두 state가 latent에서는 멀어짐

**그림 직관**:

```
raw proprio space:                          TLDR latent space:

   home/end                                     home -→ pickup -→ deliver -→ end
   ●●●● ← 시작/끝/실패 헤매기 모두 겹침         ●·············●·············●·············●
   workspace                                    ↑
   ●●  ●●  ●● ← 작업 중                        같은 raw였던 시작/끝이 latent의 다른 끝에 분리
                                                실패는 이 곡선에서 떨어져 나감

   KDE: 어느 cloud에 있어도 success/failure    KDE: progress curve의 어디인지 → 거기서
        분포가 거의 겹쳐서 log-ratio ≈ 0            success/failure 밀도 깔끔히 비교됨
```

**핵심 한 줄**: raw에서는 "겉모습(pose)"만 보고, TLDR에서는 "여정의 어느 단계"까지 봄.

#### 이유 3: Bayes-optimal score가 의미있으려면 z가 success/failure를 가를 수 있어야

CPD score `r(z) = log f+(z) − log f−(z)`는 Bayes' rule에서 나오는 log-likelihood ratio. 수식적으로는 ideal classifier. 그러나 이게 작동하려면 z에 success/failure를 가르는 정보가 들어있어야 함. 안 그러면 어떤 KDE를 써도 두 분포가 겹쳐서 r ≈ 0.

**구체적 예시**: task00 = "put both the alphabet soup and the tomato sauce in the basket"

성공/실패의 차이는 보통:
- 첫 번째 물건은 잘 옮겼는데 두 번째에서 슬립 → failure
- 둘 다 잘 옮김 → success

step 100쯤에서 success와 failure trajectory의 raw proprio를 비교하면:
- 두 trajectory 모두 두 번째 물건 근처에서 작업 중 — robot pose가 거의 똑같음
- raw 공간에서는 success-cloud와 failure-cloud가 step 100 부근에서 서로 겹침 → log-ratio ≈ 0 → 정보 없음

**TLDR이 이걸 어떻게 해결하나**:
- TLDR이 학습한 것 = "이 state가 task의 어느 진행 단계에 해당하는가"
- step 100이라도 trajectory가 향하는 방향(미세한 자세/관성 차이)이 latent에서 분리된 위치를 가지게 됨
- demo(모두 성공)와의 비교가 latent에서는 "progress curve가 demo 곡선을 따라가는가"가 됨
- failure trajectory는 demo와 같은 progress curve를 그리지 못함 → latent에서 demo 분포와 멀어짐 → KDE가 분리

**핵심 한 줄**: log-ratio라는 수식 자체는 항상 계산 가능. 그러나 **z가 success/failure 정보를 담고 있을 때만** 그 값이 의미 있는 점수가 됨. raw proprio는 그 정보를 압축적으로 담지 못하지만 TLDR latent는 (triplet loss 덕분에) 담고 있음.

### 학습 파이프라인 (한 번만 실행)

```bash
# 1. LIBERO hdf5 demo → proprio pickle
.venv/bin/python scripts/extract_libero_proprio.py \
    --suite libero_10 --out data/tldr_demos.pkl

# 2. triplet contrastive 학습
.venv/bin/python scripts/train_tldr.py
# → checkpoints/tldr.pt
```

이후 모든 inference는 `TLDREncoder.load("checkpoints/tldr.pt")`로 frozen encoder를 불러쓰는 구조 (재학습 X).

### Paper 위치

paper plan **§3.3 (φ encoder)** 에서 정의되고, **§6.4 ablation #2** 에서 QRL / HILP와 비교 — TLDR이 default.

---

## 3. 왜 proprio 8차원을 latent 64차원으로 늘리는가

config (`configs/encoder/tldr.yaml`):

```yaml
state_dim: 8
latent_dim: 64
hidden_dim: 128
num_layers: 3
```

### 한 줄로

> "8차원으로는 triplet loss를 풀 자유도가 부족하다. 64차원은 contrastive 학습의 convention + 충분한 자유도 사이의 안전한 default."

### 이유 5가지 (영향력 큰 순서)

**1. Triplet loss는 "충분한 방향"이 필요함**

config 보면 `k_pos=2, K_neg=20`. 한 anchor당 22개의 다른 점들과 비교하면서 margin 1.0으로 밀어내야 함.

8차원에서는 한 점 주위로 "서로 직교에 가까운 방향"이 최대 8개. K_neg=20개의 negative를 각각 다른 방향으로 밀어내려면 직교성이 부족함 → 한 negative를 밀면 다른 negative가 따라옴 → loss가 잘 안 내려감.

64차원이면 직교 방향이 64개 → 20개 negative를 모두 다른 방향으로 펼쳐낼 여유 충분.

**2. 임베딩 가능 점 수 (Johnson-Lindenstrauss 직관)**

500 demo × ~280 step ≈ **140K 개의 state 점**을 latent space에 distinguishable하게 박아야 함. JL lemma에 따르면 N개 점을 (1±ε) 왜곡으로 임베딩하려면 dim ≥ O(log N / ε²). log₂(140K) ≈ 17차원이 이론 하한, 실제론 마진 잡아서 ×3~4 → **50~70차원**. 64는 그 안에 정확히 들어옴.

**3. Nonlinear manifold "펼치기"**

raw proprio 8차원은 물리 제약 (gripper open/close가 2-bit-like, joint angle은 주기성 등)으로 manifold가 휘어 있음. Triplet loss는 latent에서 **Euclidean distance ≈ temporal distance**를 요구 — 휜 manifold를 곧게 펴려면 더 높은 차원에 임베딩해서 풀어야 함 (Whitney's theorem 직관: d-차원 manifold는 2d-차원에 isometric 임베딩 가능).

**4. Contrastive learning convention**

SimCLR/MoCo/SimSiam 다 128~512차원 projection head 씀. QRL/HILP 원논문도 64~256 range. 우리 64는 그 가장 보수적인 끝. "이 정도면 reviewer가 차원 선택 트집 잡지 않음"이라는 안전선.

**5. Hidden_dim=128 vs latent_dim=64의 균형**

MLP가 `(8 → 128 → 128 → 128 → 64)` 구조. 마지막 projection이 128→64. 만약 latent=8로 했으면 128→8 압축이 너무 가파름 — 정보 병목이 됨. 64는 hidden의 절반으로 자연스러운 비율.

### 솔직한 trade-off (paper에 ablation 들어가야 할 부분)

**Curse of dimensionality vs KDE downstream**

KDE는 차원이 높아지면 망가짐. Silverman bandwidth:

```
h ∝ N^(-1/(d+4))    # d=차원
```

- d=8, N=56K (점 개수):  h ∝ 56000^(-1/12) ≈ 0.40
- d=64, N=56K:           h ∝ 56000^(-1/68) ≈ 0.86 (거의 N에 무감각)

즉 **d=64에서는 더 많은 데이터를 모아도 KDE bandwidth가 안 줄어듦**. 사실상 soft nearest-neighbor 쿼리에 가까워짐. 이게 paper plan §10 risk 항목에 있어야 할 솔직한 한계.

**그럼에도 64를 쓰는 이유**: encoder가 잘 학습되면 latent space가 "task progress" 한 방향으로 길게 늘어남 — 효과적인 intrinsic dimension은 1~5차원. ambient 64차원이라도 KDE는 그 저차원 manifold 따라서 작동. 실제로 Exp1에서 KDE가 잘 분리하는 것이 그 증거.

### Ablation candidate (paper §6.4)

논문 reviewer가 100% 물어볼 질문. 미리 ablation 돌려둘 것 권장:

| latent_dim | 예상 결과 |
|---:|---|
| 8 | triplet loss 수렴 안 함 / under-fit. F1 떨어질 가능성 |
| 16 | acceptable, marginal |
| 32 | 거의 64만큼 좋을 것 |
| **64 (default)** | 본 실험값 |
| 128 | 비슷하거나 약간 향상, 그러나 KDE bandwidth 더 악화 |
| 256 | 학습은 안정적, KDE는 noisy해질 가능성 |

Sweet spot은 32~64 범위일 것. 만약 32에서 64만큼 나오면 32로 줄여서 KDE 측면에서도 정직하게 가져갈 수 있음.

### 한 줄 요약 (paper 본문용)

> "We use latent_dim=64 following standard contrastive-learning convention (SimCLR/QRL), which provides sufficient orthogonal directions for K_neg=20 negatives per anchor while remaining low enough that downstream KDE remains tractable on the learned task-progress manifold."

---

## 관련 artifact

- `reports/exp2/summary.md` — Exp2 main 결과 (F1 vs N sweep)
- `reports/exp2/explain_critical_phase.png` — §1의 3-episode 비교 그림
- `reports/exp2/sweep/loocv_f1_vs_n.png` — F1-vs-N 메인 figure
- `reports/exp2/sweep/loocv_f1_vs_n.json` — 수치 데이터
- `reports/exp2/sweep/per_traj.json` — episode별 trajectory-level feature
- `scripts/sweep_cpd_n.py` / `scripts/sweep_cpd_n_loocv.py` — 실험 스크립트
- `src/cpd/encoders/tldr.py` / `configs/encoder/tldr.yaml` — TLDR 정의
