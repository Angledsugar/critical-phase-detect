# TLDR contrastive 학습 — "두 state 의 latent 거리가 시간 거리에 비례하도록"

> 이 한 문장이 정확히 무엇을 의미하는지, 왜 contrastive 학습이라는 도구를 쓰는지,
> 학습이 끝나면 latent space 가 어떤 모양이 되는지를 풀어서 정리.

---

## 1. 한 문장의 의미를 풀어쓰기

다음 두 가지가 같은 값이 되도록 만들고 싶다:

```
   "두 state s_a, s_b 가  φ 를 통과한 후의  latent 거리"

   ↕  (비례하도록)

   "두 state s_a, s_b 의  시간상 거리"
```

즉:

```
   ||φ(s_a) − φ(s_b)||      ∝      | t_a − t_b |
   ─────────────────              ──────────────
       latent 거리                    시간 거리
```

여기서 $s_a$ 는 어떤 demo 의 $t_a$ 시점 state, $s_b$ 는 같은 demo 의 $t_b$ 시점 state.

직관 한 줄:

> **"두 state 가 demo 안에서 1 step 떨어져 있으면 latent 에서도 1 단위만,
>   100 step 떨어져 있으면 latent 에서도 100 단위 정도 떨어지도록 인코더를 학습"**

---

## 2. "latent 거리" 가 무엇인가

State $s \in \mathbb{R}^8$ (LIBERO proprio: 3 pos + 3 ori + 2 gripper) 를 받아서
$d = 64$ 차원 벡터로 보내는 함수 $\varphi$ 가 있다고 하자 (MLP).

```
   s_a ∈ R^8          φ          φ(s_a) ∈ R^64
   ───────  →  ───────────  →    ──────────────
   raw state    encoder (MLP)     latent vector
```

두 state 의 latent 거리:

```
   d_latent(s_a, s_b)  =  || φ(s_a) − φ(s_b) ||_2
                       =  sqrt( Σ_{k=1..64} ( φ(s_a)_k − φ(s_b)_k )² )
```

→ 64차원 공간에서의 Euclidean 거리. 그냥 두 벡터 빼고 노름.

---

## 3. "시간 거리" 가 무엇인가

같은 demo $\tau = (s_0, s_1, \ldots, s_T)$ 에서:

```
   d_time(s_a, s_b)  =  | t_a − t_b |        # 단위: step
```

예시:
- $s_5, s_7$ → 시간 거리 2 step
- $s_0, s_{200}$ → 시간 거리 200 step
- $s_0, s_T$ → 시간 거리 $T$ step (demo 끝까지)

```
demo τ :   ●─●─●─●─●─ ... ─●─●─●─●─●
           ↑       ↑                ↑
           s_0     s_5              s_T
           t=0     t=5              t=T
```

---

## 4. "비례한다" 는 게 정확히 무슨 뜻인가

엄밀한 의미와 실용적 의미가 살짝 다르다.

### 4.1 엄밀한 (이상적) 의미

```
   || φ(s_a) − φ(s_b) ||  =  c · | t_a − t_b |        ( c > 0 상수 )
```

즉 latent 거리 = (상수) × 시간 거리. 만약 $c = 1$ 이면 "1 step 떨어진 두 state 는
latent 에서 1 단위 떨어지고, 100 step 떨어진 두 state 는 100 단위 떨어진다".

### 4.2 실제 학습 목표 (느슨한 의미)

위의 엄밀한 등식을 그대로 강제하는 건 어려움 (5.1 참조). 대신 **순서 (ordering)** 만 보장:

```
   if  | t_a − t_p | < | t_a − t_n |
   then  || φ(s_a) − φ(s_p) || < || φ(s_a) − φ(s_n) ||
```

즉 "시간상 가까운 쌍은 latent 에서도 더 가까워야 한다". 비례 상수까지 학습하진 않고,
**monotonic 관계 (시간 멀수록 latent 도 멀어짐)** 만 학습.

실용적으로는 margin $m$ 과 hinge loss 때문에 "1 step ≈ 1 단위" 근처로 수렴함
(우리 학습에서는 학습 종료 시 인접 step latent 거리 ≈ 1, 끝–처음 latent 거리 ≈ 70).

### 4.3 "두 거리가 같다" — 구체적 숫자로 다시 보기

핵심 의문: 무엇과 무엇이 같다는 건가?

```
   LHS  :  || φ(s_a) − φ(s_b) ||            ← 64차원 latent space 의 Euclidean 거리
   RHS  :  | t_a − t_b |                     ← 같은 demo 안에서 두 시점의 step 차이

   "같다"  ⟺  두 scalar 값이 비슷한 크기를 가짐
```

**두 값 다 단일 양의 실수**:

| 항 | 입력 | 처리 | 출력 | 단위 |
|---|---|---|---|---|
| LHS (latent 거리) | $s_a, s_b \in \mathbb{R}^8$ | $\varphi$ 통과 후 차의 norm | $\in \mathbb{R}_+$ | latent 의 임의 단위 |
| RHS (시간 거리)   | $t_a, t_b \in \mathbb{Z}$ (같은 demo) | 차의 절댓값 | $\in \mathbb{Z}_+$ | step (= 1 control timestep) |

**우리 실험에서 실제 학습된 값**:

LIBERO-Long demo 500 개로 학습한 $\varphi$ 의 학습 종료 시:

```
   학습 종료 d_an / d_ap ≈ 70

   d_ap (anchor–positive 거리²) ≈ 1 단위²
       시간상 가까운 쌍 (예: 1–2 step 차이) 의 latent 거리² 평균
       → latent 거리 ≈ 1 단위

   d_an (anchor–negative 거리²) ≈ 70 단위²
       시간상 먼 쌍 (예: 평균 70 step 차이) 의 latent 거리² 평균
       → latent 거리 ≈ √70 ≈ 8.4 단위
```

이걸 풀어서 표로 보면:

| 같은 demo 안에서 | 시간 거리 | latent 거리 (관찰) |
|---|---|---|
| 인접 step       | 1 step  | ≈ 1 단위 |
| 가까운 step 쌍   | 1–5 step | ≈ 1–2 단위 |
| 먼 step 쌍       | ≈ 70 step | ≈ 8 단위 |
| 끝–처음          | demo 길이 (≈ 100 step) | ≈ 70 단위 |

즉 "1 step ≈ 1 단위" 의 **느슨한** 비례. 엄밀한 등식 ($c \cdot |t_a - t_b|$) 은 아니지만, **monotonic 관계** + **roughly 동일 scale** 은 유지됨.

### 4.4 "같다" 가 주는 의미 — latent space 에서 거리를 잰다는 것의 새 해석

학습 후 latent 거리는 더 이상 임의의 64차원 metric 이 아님. 다음과 같이 해석 가능:

```
   "latent 에서 두 state 가 r 단위 떨어져 있다"
                    ↕
   "이 두 state 는 task 시간축에서 약 r step 차이의 동작"
```

**구체적 예시**:

- ep115 의 step 50 과 step 60 이 latent 에서 9 단위 떨어져 있다면
  → "약 9 step 차이의 task progress" → 정상.
- ep118 의 step 50 과 step 60 이 latent 에서 0.5 단위 떨어져 있다면
  → "거의 같은 task progress" → 로봇이 정체 중 (1 step 가야 할 거리를 0.1 단위만 감)
  → 이게 § 4.4 의 critical phase 신호의 출발점.
- ep115 의 step 50 과 ep118 의 step 50 이 latent 에서 30 단위 떨어져 있다면
  → "두 episode 가 같은 step 시점인데 task progress 가 매우 다름"
  → 한 쪽이 막혀 있다는 신호.

→ **latent 거리 = task progress 차이의 proxy**. 이게 raw proprio Euclidean 으로는 안 되는 (`explain.md` § 2 의 단위 문제 + § 11.1 의 progress vs raw time 구분), TLDR 학습 후에만 얻어지는 성질.

### 4.5 왜 이게 KDE 와 G2 에 핵심인가

**KDE** (§ 4.3-4.4): 두 latent 점 $z_a, z_b$ 의 거리 $\|z_a - z_b\|$ 가 kernel weight 를 결정.
- "거리 = task progress 차이" 이므로 kernel weight 가 "task 의 같은 phase 끼리만 영향"을 의미.
- KDE 가 task phase 별 빈도를 측정하는 것과 동등.

**G2** (§ 4.2): $\|\varphi(s_T) - g\| \leq \varepsilon$
- "거리 = task progress 차이" 이므로 이 조건은 "rollout endpoint 의 task progress 가 demo endpoint 의 task progress 와 ε step 이내" 와 동등.
- 즉 "task 가 거의 끝났는가" 를 직접 측정.

→ "latent 거리 ≈ 시간 거리" 가 학습되지 않으면 KDE / G2 모두 의미 없는 metric 위에서 작동하게 됨. TLDR 의 **유일한 직접 학습 목표**이지만, 그게 무너지면 paper 의 § 4.2–4.4 가 다 무너짐.

---

## 5. 왜 "contrastive" 인가 — 직접 회귀가 안 되는 이유

### 5.1 직접 회귀 방식 (안 쓰는 이유)

가장 직관적인 방식은 supervised regression:

```
   min_φ  E_{s_a, s_b} [ ( || φ(s_a) − φ(s_b) ||  −  | t_a − t_b | )² ]
```

"latent 거리를 시간 거리에 직접 회귀". 그러나 문제점:

1. **Target 값의 absolute scale 이 임의적**
   "1 step = latent 1 단위" 라는 단위 자체가 자의적. 다른 demo 가 다른 속도로 진행되면
   같은 1 step 도 다른 의미 ("멈춰있는 step" vs "빠르게 움직이는 step").
2. **Latent space 가 collapse**
   φ 가 모든 state 를 한 점에 매핑하면 latent 거리는 항상 0 → loss 가 step 거리를
   직접 따라가야 함 → 매우 불안정.
3. **Outlier 에 약함**
   demo 사이의 시간 비교 ($\tau^{(i)}$ 의 step 5 vs $\tau^{(j)}$ 의 step 5) 가 의미
   불명. 같은 step index 라도 task 진행도가 다를 수 있음.

### 5.2 Contrastive 방식 (쓰는 이유)

직접 절댓값 대신 **쌍 비교** 로 가르침:

```
   "이 두 쌍 중에서, 어느 쪽이 시간상 더 가까운가? → 그 쪽이 latent 에서도 더 가까워야"
```

이렇게 하면:

- absolute scale 문제 사라짐 (ordering 만 학습).
- collapse 회피 (negative pair 가 멀어지도록 push, 항상 spread).
- demo 간 비교 안 함 (positive / negative 둘 다 **같은 demo 내부**에서 뽑음).

→ 이것이 **contrastive (비교 기반) 학습**. 회귀가 아니라 분류 비슷한 방식.

---

## 6. Triplet loss — contrastive 의 구체적 구현

### 6.1 한 step 의 학습

매 step 마다 다음을 수행:

```
(1) demo 한 개 τ^(i) 뽑기
(2) 시간 index 세 개 뽑기 :  t_a, t_p, t_n     단,  |t_a − t_p| < |t_a − t_n|
(3) 세 state 의 latent 계산  :  φ(s_ta), φ(s_tp), φ(s_tn)
(4) loss 계산 :
        d_ap = || φ(s_ta) − φ(s_tp) ||²
        d_an = || φ(s_ta) − φ(s_tn) ||²
        L = max( 0,  d_ap − d_an + m )
(5) gradient descent
```

### 6.2 loss 의 의미

```
L = max( 0,   d_ap  −  d_an  +  m  )
```

- $d_{ap} - d_{an} + m \leq 0$, 즉 $d_{ap} + m \leq d_{an}$  → loss = 0 (이미 잘 학습됨)
- $d_{ap} + m > d_{an}$  → loss > 0 (positive 가 negative 보다 멀거나, margin 이내)

**margin $m$ 의 역할**: positive 와 negative 사이에 최소 간극 $m$ 을 보장.
$d_{ap} < d_{an}$ 만으로는 부족하고, **확실히** 더 가까워야 (적어도 $m$ 만큼) loss = 0.

```
   loss
    │
    │\
    │ \
    │  \                margin 영역  →  0 으로 가는 hinge
    │   \
    │    \____________
    │
    └──────────────────────────  d_an − d_ap
              m
```

### 6.3 hyperparameter (paper 실험값)

```
   k_pos = 2          한 anchor 당 positive 후보 수 (안에서 가까운 시점)
   K_neg = 20         한 anchor 당 negative 후보 수 (멀리 떨어진 시점)
   m     = 1.0        margin
   epoch = 50
   batch = 50  (demo 단위)
   triplet/step = 256
```

→ 한 epoch 에 ≈ 50 batch × 256 triplet = 12,800 triplet, 50 epoch ⇒ 64만 triplet 학습.

---

## 7. 학습 후 latent space 가 어떻게 되는가

### 7.1 학습 전 (random φ)

```
   latent space :   ● ●  ●     ●
                       ●  ●  ●   ●         ← 모든 state 가 random 위치
                  ●   ●    ●  ●
                       ● ●     ●
```

→ 시간 거리와 latent 거리에 아무 관계 없음.

### 7.2 학습 후 (수렴한 φ)

한 demo 의 state 들이 시간 순서대로 latent 의 한 path 를 따라 흐름:

```
                  s_T
                   ●
                    ●
                     ●
                      ●
                       ●
                       ●         ← demo 의 trajectory
                        ●         가  latent 의 곡선으로
                         ●        펴짐
                         ●
                        ●
                       ●
                      ●
                     ●
                    ●
                   ●
                  s_0
```

여러 demo 가 같은 task 면 비슷한 path 를 그리고 (start 영역에 모이고 end 영역에 모임):

```
                       ::: end cluster  (모든 demo 의 φ(s_T))
                      ::
                     ::
                    ::
                   ::         ← 여러 demo 가
                  ::             같은 path 를
                 ::              따라 흐름
                ::
              :::: start cluster (모든 demo 의 φ(s_0))
```

이 모양이 §4.2 의 G2 labeler 를 가능하게 함 (`explain_g2_bridge.md` 참조).

### 7.3 우리 실제 학습 결과 (paper §4.1)

```
   학습 시작        →    종료
   ─────────────────────────────
   violation 26 %  →    0.5 %       # margin 위반하는 triplet 비율
   d_an / d_ap  ≈ 1 →   ≈ 70        # negative 가 positive 의 70 배 멀음
```

즉 학습 끝나면 "시간상 먼 쌍" 이 "시간상 가까운 쌍" 보다 **latent 에서 70 배 더
멀다**. 이게 충분한 ordering 신호 → §4.4 의 KDE log-ratio 가 의미를 가짐.

---

## 8. 왜 이게 critical phase detection 에 유용한가

학습 끝난 φ 는 다음과 같은 **부산물 (downstream) property** 를 가짐:

| Property | 의미 | 어디에 쓰이는가 |
|---|---|---|
| **시간 단조성** | latent 좌표가 task 진행도 (0→1) 와 monotonic | §4.4 KDE 의 신호원 |
| **start cluster** | 모든 demo 의 $\varphi(s_0)$ 가 한 영역 | §4.2 라벨러의 "시작 영역" |
| **end cluster** | 모든 demo 의 $\varphi(s_T)$ 가 한 영역 | §4.2 라벨러의 "goal 영역" $g$ |
| **task 분리** | 실패 trajectory 는 latent path 의 "중간 어디" 에 멈춤 | §4.2 의 succ/fail 분리, §4.4 의 $r_t < 0$ |
| **단위 정규화** | 모든 차원이 task-relevant 신호로 정렬 | raw proprio 의 회전/그리퍼 단위 문제 해소 |

**중요**: 이 property 들은 **명시적 학습 목표가 아니라 부산물**. triplet loss 는 단지
"시간 거리 ordering 보존" 만 강제했는데, MLP 의 함수성 + demo 의 물리적 유사성과
결합되어 자동으로 이런 구조가 생김.

---

## 9. 비유 — 마라톤 트랙

42 km 트랙 위에서 100 m 마다 사진을 찍는다고 가정.

- **raw state** = 사진의 pixel array. 같은 위치 (예: 21 km 지점) 라도 광원·복장에
  따라 pixel 이 완전히 다름.
- **시간 거리** = 두 사진의 km 차이 (트랙 위치 차이).
- **latent 거리** = 학습된 encoder 가 사진을 코딩한 후의 벡터 거리.
- **TLDR contrastive 학습** = "비슷한 km 의 사진끼리는 latent 가 가깝게, 멀리 떨어진
  km 의 사진끼리는 멀게 인코딩" 을 학습. 직접 km 를 회귀하지 않고, "두 쌍 중 어느
  쪽이 더 가까운가?" 만 비교해서 학습.
- **학습 후의 latent space** = 트랙의 km 좌표 (0 → 42) 를 그대로 코딩한 1차원
  manifold 비슷한 모양. start (0 km) 는 한 영역, finish (42 km) 는 다른 영역.

여러 명의 marathoner (= 여러 demo) 가 비슷한 트랙을 달리면, 그들 모두 같은 latent
path 를 그림. start 영역과 finish 영역에는 모두가 모임. 이게 §4.2 가 "finish 영역에
도착했는가?" 로 성공 판정할 수 있는 근거.

---

## 10. 한 줄 요약 (paper notation 정의용)

> **TLDR contrastive 학습** = "두 state 의 latent 거리 순서가 그들의 시간 거리
> 순서와 일치하도록 encoder $\varphi$ 를 triplet hinge loss 로 학습한다.
> 결과적으로 latent space 에서 demo 의 trajectory 가 시간 순서대로 흐르는 매끄러운
> path 로 펴지고, demo 간 start / end 영역이 자동으로 cluster 된다."

---

## 11. Demo 마다 성공까지 걸린 step 수가 다른 경우 — 어떻게 표현되는가

**의문**: 같은 task 의 demo 라도 길이가 다양함.

```
demo i :  ●─●─●─●─●─●─●─●─●─●         T_i = 100 step  (빠른 demo)
demo j :  ●─●─●─●─●─●─●─●─●─●─●─●─●   T_j = 300 step  (느린 demo)
```

triplet loss 가 "시간 거리 큰 쌍 → latent 거리 커야" 라고 학습한다면:

- demo $i$ 의 (step 0, step 100) 의 latent 거리 ≈ "100 단위"
- demo $j$ 의 (step 0, step 300) 의 latent 거리 ≈ "300 단위"

→ 두 demo 의 endpoint 가 latent 의 **서로 다른 위치** 에 갈 것 같음.
이러면 § 4.2 가 의지하는 "endpoint cluster (P1)" 이 깨짐.

그런데 실제로는 안 깨진다. 왜?

### 11.1 짧은 답

> **TLDR latent 는 "raw time" 이 아니라 "physical progress (= task 의 어느 단계인가)"
> 를 인코딩한다.** demo 길이가 달라도 시작·끝의 physical state (proprio) 가 같으면
> latent 의 시작·끝도 같다.

### 11.2 두 메커니즘이 충돌하는 지점

학습 시 다음 두 제약이 동시에 작용:

```
(A) Triplet loss (soft signal)
    "같은 demo 안에서 시간 거리 큰 쌍 → latent 거리 큰 쌍"
    → demo j 의 step 0 와 step 300 을 latent 에서 멀게 하려 함

(B) MLP 함수성 (hard constraint)
    "같은 input → 같은 output"
    → φ 는 demo id 를 입력으로 받지 않고 proprio 만 받으므로,
      같은 proprio 는 항상 같은 latent 로 매핑됨
```

이제 시나리오를 보자:

- demo $i$ (빠름, $T_i = 100$) 의 step 25 → 로봇이 "object 위에 도착" (physical progress 25%)
- demo $j$ (느림, $T_j = 300$) 의 step 75 → 로봇이 "object 위에 도착" (physical progress 25%)

두 state 의 **proprio 값이 거의 같다** (둘 다 같은 task 의 같은 단계).

```
(B) 가 강제 :  φ(s_25 of demo i)  ≈  φ(s_75 of demo j)
```

그런데 시간 거리는 다름:
- demo $i$ 에서 step 0 → step 25 는 **25 step** 떨어짐
- demo $j$ 에서 step 0 → step 75 는 **75 step** 떨어짐

(A) 의 triplet loss 가 demo $j$ 의 step 0–75 를 demo $i$ 의 step 0–25 보다 멀게 만들려고 하면, (B) 와 충돌.

### 11.3 어떻게 해결되는가

학습은 **(B) 가 우세하게** 수렴함. 이유:

1. **(B) 는 hard constraint** : MLP 가 함수인 한 깰 수 없음.
2. **(A) 는 soft constraint** : triplet loss 는 gradient-based, 그리고 ordering 만
   강제 (절대 거리는 안 강제). margin $m$ 이내에서 유연성 있음.
3. **Triplet loss 는 같은 demo 안에서만 비교** : demo $i$ 와 demo $j$ 사이의
   "step 25 vs step 75" 를 직접 비교하는 triplet 은 학습 데이터에 없음. 그러므로
   (A) 가 cross-demo 충돌을 강제할 수단이 없음.

결과: 학습이 수렴하면 latent path 가 **physical progress 좌표** 로 정렬됨.

```
   "demo i 의 step 25" 와 "demo j 의 step 75" → 같은 latent 위치 (둘 다 progress 25%)
   "demo i 의 step 100" 과 "demo j 의 step 300" → 같은 latent 위치 (둘 다 progress 100%)
```

### 11.4 시각적 직관

같은 latent path 위에 두 demo 가 놓이지만, **속도가 다르다**:

```
latent space (한 방향 path 로 simplification) :

        progress 0%      25%      50%      75%      100%
          │              │         │         │         │
   path:  ●──────────────●─────────●─────────●─────────●
          │                                            │
          │                                            │
   demo i :   start    s_25     s_50     s_75      s_T=100    ← 한 step 당
   ───────────────────────────────────────────────             1% progress
   step 수:    0        25       50       75        100        (빠른 demo)
                                                              

   demo j :   start              s_75            s_225  s_T=300  ← 한 step 당
   ────────────────────────────────────────────────────────       0.33% progress
   step 수:    0                  75              225    300       (느린 demo)
```

두 demo 모두 **같은 latent path 위를 따라간다**. 차이는 **얼마나 많은 step 을 같은
구간에 머무르는가**:

- 빠른 demo: progress 25% 에서 25% 까지 가는 데 25 step.
- 느린 demo: 같은 구간을 가는 데 75 step. **같은 latent 영역에 3배 더 머무름.**

핵심: **endpoint 는 두 demo 모두 progress 100% 영역 (같은 latent cluster)**.

### 11.5 왜 이게 § 4.2 G2 labeler 에 좋은가

성공 demo 가 빠르든 (T=100) 느리든 (T=300) endpoint cluster 가 같음:

```
   demo i (빠른 성공) :  φ(s_T_i)  →  end cluster
   demo j (느린 성공) :  φ(s_T_j)  →  end cluster        ← 같은 영역
   demo k (실패) :      φ(s_T_k)  →  중간 어디 (progress 60% 에서 멈춤)
```

→ § 4.2 의 "$\varepsilon$-ball 안에 들어왔는가?" 가 demo 길이에 **무관하게** 작동.

만약 TLDR 이 raw time 을 인코딩했다면, 빠른 demo 와 느린 demo 의 endpoint 가
latent 의 다른 곳 (예: latent 100 vs latent 300 단위) 에 있어 cluster 가 깨졌을 것.

### 11.6 왜 이게 § 4.4 critical scoring 에 좋은가

failure rollout 이 "approach 단계" 에서 200 step 정체한다고 하자:

```
   step 0  10  20  30  ...        200  210  ...  520
            └────────────────────┘
                progress 25% 영역에서 200 step 정체
```

latent 에서도 200 step 동안 **같은 영역** 에 머무름 (progress 정체 ⇒ latent 정체).

이게 § 4.4 의 KDE 에서:
- 그 영역의 $\tilde{f}_+$ (success buffer density) 는 낮음 — 성공 demo 는 이 영역을
  빠르게 통과하므로 머무는 step 이 적음.
- 그 영역의 $\tilde{f}_-$ (failure buffer density) 는 높음 — 실패 rollout 이 자주
  여기서 정체.
- → $r_t = \log \tilde{f}_+ - \log \tilde{f}_- < 0$ → critical step 으로 잡힘.
- → $n_\text{crit\_steps}$ 가 200 으로 커짐.

즉 **"physical progress 정체"** 가 자연스럽게 critical phase 신호가 됨.
시간 정체와 latent 정체가 같은 의미를 가지는 게 아니라, **progress 정체** 가 latent
정체로 이어지는 것.

### 11.7 한계 / Edge case

1. **반복 동작 (loop) 이 있는 task** : 예를 들어 demo 가 "wipe table 10 times" 같은
   반복 작업이면, 같은 physical config 가 여러 번 등장. MLP 함수성 때문에 모두 같은
   latent 에 매핑됨 → triplet loss 가 깨짐 (1번째 wipe 와 10번째 wipe 가 latent 에서
   같음). LIBERO 의 monotonic task 에서는 문제 없음.

2. **Demo 가 같은 단계에서 머무는 길이가 매우 다른 경우** : 예를 들어 demo $i$ 가
   "approach 단계" 에 10 step 만, demo $j$ 가 1000 step 머무름. 둘의 endpoint cluster
   는 여전히 같지만, 중간 latent 의 데이터 분포가 매우 unbalanced 가 됨 → KDE
   bandwidth 추정에 영향. paper § 5.4 의 N=200 보다 더 많은 demo 가 필요할 수 있음.

3. **Physical state 가 같아도 "task 가 끝났는가" 가 다른 경우** : 예를 들어 로봇이
   home pose 로 복귀 → physical state 는 home, latent 는 "progress 0%" 영역. 그런데
   task 자체는 끝났을 수도. raw proprio 만 보는 G2 는 이 케이스를 구분 못 함
   (`explain_g2_bridge.md` § 8 참조).

### 11.8 한 줄 요약 (§ 11 결론)

> "TLDR 은 시간 (raw step 수) 이 아니라 **physical progress (task 의 어느 단계인가)**
> 를 인코딩한다. demo 길이가 달라도 시작·끝 physical state 가 같으면 latent 의
> 시작·끝도 같다. 빠른 demo 와 느린 demo 는 같은 latent path 위를 다른 속도로
> 지나갈 뿐, endpoint 는 같은 cluster 에 도달한다.
> 이 progress 좌표 성질이 (i) demo 길이 무관한 G2 labeling 과 (ii) physical
> progress 정체로서의 critical phase 검출을 가능하게 한다."
