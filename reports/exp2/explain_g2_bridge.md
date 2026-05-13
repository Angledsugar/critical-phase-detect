# §4.1 → §4.2 의 매끄러운 연결

> Paper § 4.1 (TLDR encoder) 에서는 trajectory 전체를 학습 신호로 쓰는데, § 4.2 (G2 labeler)
> 에서는 갑자기 마지막 state $\varphi(s_T)$ 한 점만 봐서 라벨링한다.
> 두 절 사이가 점프되는 것처럼 보이는 이유와, 실제로는 자동으로 연결된다는 점을 정리.

---

## 1. 점프되는 지점이 정확히 어디인가

| 절 | 학습 / 사용 대상 | 가정 |
|---|---|---|
| § 4.1 | trajectory 전체 (anchor / pos / neg 가 시간상 펼쳐짐) | "시간상 가까우면 latent 거리 가깝게" |
| § 4.2 | 마지막 state $\varphi(s_T)$ 한 점 | "성공 endpoint 들이 latent 한 영역에 모여 있음" |

§ 4.1 의 학습 목표 ("시간 거리") 와 § 4.2 의 라벨링 가정 ("endpoint clustering") 은
**겉으로 보면 서로 다른 성질**이다. 둘이 어떻게 연결되는지 명시하지 않으면,
§ 4.2 가 § 4.1 과 무관한 가정을 새로 도입하는 것처럼 보인다.

→ **연결고리: 시간 좌표를 학습한 결과로 endpoint clustering 이 자동으로 따라온다.**

---

## 2. § 4.1 이 만들어내는 latent space 의 모양

### 2.1 Notation

학습 데이터: $N = 500$ 개의 demo trajectory.

```
demo i :  τ^(i) = ( s_0^(i), s_1^(i), ..., s_{T_i}^(i) )

    s_t^(i) ∈ R^8      i 번째 demo 의 t-번째 step 의 raw proprio state
                         (3 pos + 3 ori + 2 gripper, LIBERO)
    T_i               i 번째 demo 의 길이 (step 수)
    φ : R^8 → R^64    학습 대상인 MLP encoder (TLDR)
```

매 학습 step 에서 한 demo $\tau^{(i)}$ 를 고르고, 그 안에서 시간 index 3 개를 뽑음:

```
t_a ∈ {0, 1, ..., T_i}      anchor    의 시간 index
t_p ∈ {0, 1, ..., T_i}      positive  의 시간 index
t_n ∈ {0, 1, ..., T_i}      negative  의 시간 index

s_ta := s_{t_a}^(i)         anchor    state   (시점 t_a 의 proprio)
s_tp := s_{t_p}^(i)         positive  state   (anchor 와 시간상 가까운 쪽)
s_tn := s_{t_n}^(i)         negative  state   (anchor 와 시간상 먼 쪽)

샘플링 조건 :  | t_a − t_p |  <  | t_a − t_n |
              (positive 가 anchor 와 시간상 더 가까워야 함)

m ∈ R_+                     margin 하이퍼파라미터 (논문 실험에서 m = 1.0)
```

즉 anchor / positive / negative 세 점은 **같은 한 demo 의 서로 다른 세 시점**의
state 이고, positive 는 anchor 와 시간상 가까운 시점, negative 는 더 먼 시점.

### 2.2 학습 식 (Triplet contrastive loss)

위 notation 으로 한 triplet 의 loss:

```
L( s_ta, s_tp, s_tn )
    = max( 0,
            ||φ(s_ta) − φ(s_tp)||²       ← anchor–positive latent 거리²
          − ||φ(s_ta) − φ(s_tn)||²       ← anchor–negative latent 거리²
          + m )                          ← margin
```

직관: "anchor–positive 거리" 가 "anchor–negative 거리 − m" 보다 커지면 loss 발생.
→ 학습이 진행되며 anchor 와 positive 는 latent 에서 **가깝게**, anchor 와 negative
는 **m 이상 멀게** 배치됨.

기댓값으로 쓰면:

```
L_φ = E_{τ^(i),  t_a, t_p, t_n} [ L( s_{t_a}^(i), s_{t_p}^(i), s_{t_n}^(i) ) ]
```

- 셋 다 **같은 demo $\tau^{(i)}$ 내부**에서 sampling.
- demo $i$ 와 demo $j$ 사이엔 **어떤 직접 제약도 없다**.

### 2.3 그런데도 endpoint 가 cluster 되는 메커니즘

학습 끝나면 서로 다른 demo 의 $\varphi(s_T^{(i)})$, $\varphi(s_T^{(j)})$ 가
latent 에서 가까운 영역에 모이는 현상이 관찰됨. 이유:

**(a) 함수성 제약**
MLP 인코더는 **demo id 를 입력으로 받지 않는다**. 8차원 proprio 한 벡터만 받음.
같은 입력 → 같은 출력 (function). 따라서 두 demo 의 proprio 입력이 거의 같으면
latent 도 거의 같아야만 한다.

**(b) LIBERO demo 의 물리적 유사성**
- 모든 demo 가 **같은 로봇 home pose 근처에서 시작** → $s_0^{(i)} \approx s_0^{(j)}$.
- 같은 task 의 demo 가 **같은 goal config 근처에서 끝남** → $s_T^{(i)} \approx s_T^{(j)}$.

(a) + (b) 결합:
$\varphi(s_0^{(i)}) \approx \varphi(s_0^{(j)})$ 가 자동.
$\varphi(s_T^{(i)}) \approx \varphi(s_T^{(j)})$ 도 자동.

**(c) Triplet loss 의 강제**
"시작과 끝은 멀어야 한다" 를 모든 demo 에서 동시에 만족시켜야 함.
MLP 가 만들 수 있는 가장 간결한 해:

> **"task 진행도 (0→1) 에 따라 latent space 의 한 방향으로 흐르는 매끄러운 곡선"**

즉 모든 demo 가 latent 의 같은 path 를 따라 흐르고,
시작은 한 영역에, 끝은 다른 한 영역에 자동으로 모이게 된다.

### 2.4 결과 — 학습 후 latent space 의 모양

```
                  ┌──────────────────────────────────────────┐
                  │                                          │
                  │   . start cluster                        │
                  │   .. (모든 demo 의 φ(s_0))                │
                  │   .                                      │
                  │     `.                                   │
                  │       `.                                 │
                  │         `--- task progress 흐름 --→      │
                  │                          `.              │
                  │                            `..           │
                  │                              :: end      │
                  │                              :  cluster  │
                  │                                          │
                  └──────────────────────────────────────────┘
                                                  ↑
                                       § 4.2 가 의지하는 영역
```

이 **end cluster 가 § 4.1 의 부산물로 자동 생긴다**는 점이 § 4.1 → § 4.2 의 다리.

---

## 3. § 4.2 가 그 성질을 그대로 가져다 씀

§ 4.1 의 latent space 가 end cluster 를 가지면, success 판정은 자동.

### 3.1 Goal point

```
g = (1/N) Σ_i φ(s_T^(i))     # demo endpoint 들의 평균
```

end cluster 가 좁으면 $g$ 는 cluster 의 좋은 대표점.

### 3.2 Threshold

```
ε = Quantile_0.95 ( { ||φ(s_T^(i)) − g||  :  i = 1..N } )
```

demo endpoint 들이 실제로 흩어진 정도의 95%-반경. 가장 멀리 흩어진 demo 도
포용하되 top 5% outlier 는 제외.

### 3.3 라벨링 규칙

```
ŷ(τ) = 1[ ||φ(s_T) − g|| ≤ ε ]
```

"이 rollout 의 endpoint 가 demo cluster 안에 들어왔는가?"

### 3.4 왜 이게 합리적인가

- demo cluster 는 latent 에서 **"task 가 완료된 상태"** 영역.
- rollout 이 task 를 완료 → $\varphi(s_T)$ 가 같은 영역에 도착
  (§ 4.1 의 latent 가 그렇게 학습됐으니까).
- rollout 이 task 를 실패 → $\varphi(s_T)$ 는 그 영역에서 멈춤
  (실패 rollout 의 $s_T$ 는 시간상 **"task 중간 어디서 멈춘 상태"**
  로 인코딩됨).

즉 § 4.2 는 "마지막 state 만 보는 임의의 휴리스틱" 이 아니라,
**§ 4.1 이 만들어준 task-progress 좌표축의 끝점에서의 거리 비교**다.

---

## 4. § 4.1 을 안 거치면 어떻게 되는가 (대조)

raw proprio space 에서 동일한 G2 절차를 돌리면:

| 단계 | latent space (TLDR) | raw proprio space |
|---|---|---|
| $g = \mathrm{mean}_i \, s_T^{(i)}$ | endpoint cluster 중심 | 의미 있는 점이 아닐 수 있음 |
| $\lVert s_T − g \rVert$ | task-progress 끝점에서의 거리 | 단위 섞인 8차원 거리 |
| $\varepsilon$ ball | "성공 endpoint 영역" | 회전 성분이 dominant 한 임의 ellipsoid |

raw proprio 의 문제:

1. **단위 문제** (`explain.md` §2 에서 다룸): 8차원 unit 이 섞여 ball 이
   회전축만 봄.
2. **시작 = 끝 문제**: LIBERO 에서 robot home pose 는 task 시작 시점과
   끝난 후 시점이 거의 같음 (gripper 만 다름). raw 에서 $s_T \approx s_0$ 인
   경우가 흔해서 "성공 endpoint 가 시작 영역과 분리 안 됨".
3. **물체 정보 없음**: proprio 8차원은 로봇만 알고 물체 위치는 모름.
   그런데 task success 는 물체 상태로 결정됨.

TLDR 이 한 일: contrastive 학습으로 proprio 안에 숨어있는 **"task 진행도"**
신호 (예: arm 을 뻗었는지, gripper 를 닫았는지의 미세한 sequence) 를 증폭하여
latent 로 띄움. 그 결과 시작과 끝이 latent 에서 분리되고, end cluster 가 살아남.

---

## 5. § 4.2 가 의지하는 2 개의 성질

§ 4.2 가 작동하기 위해 latent space 가 만족해야 하는 성질:

```
(P1) Endpoint clustering :
        같은 task 의 성공 demo 의 φ(s_T) 들이 좁은 영역에 모임.

(P2) Endpoint separation :
        실패 rollout 의 φ(s_T) 는 그 영역 밖에 떨어짐.
```

| 성질 | § 4.1 이 직접 학습? | 어떻게 얻어짐 |
|---|---|---|
| P1 | ✗ (직접 목표 아님) | 함수성 + demo 물리적 유사성 + triplet loss 의 부산물 |
| P2 | ✗ | P1 의 부산물 — 실패 rollout 의 $s_T$ 는 latent 의 "중간 영역" 에 위치 |

→ § 4.2 가 마지막 state 만 봐도 되는 이유:

> "§ 4.1 이 만든 latent space 에서는 마지막 state 가 이미 task progress 의
> 끝 좌표를 인코딩하고 있고, 성공한 trajectory 는 모두 같은 끝 좌표 영역에 모인다."

trajectory 전체의 시간 정보가 § 4.1 에서 latent 의 task-progress 좌표축으로
**압축되어 마지막 점에 응축**되어 있으므로, § 4.2 가 $s_T$ 한 점만 봐도 충분.

---

## 6. 역할 분담 — labeler 와 score 가 다른 것을 봄

| 모듈 | 입력 | 출력 | 봐야 하는 것 |
|---|---|---|---|
| **G2 (§ 4.2)** | $\varphi(s_T)$ 한 점 | 1-bit 라벨 (succ/fail) | 결과 |
| **CPD score (§ 4.4)** | $\varphi(\tau)$ 전체 | 매 step 의 $r_t$ | 과정 |

trajectory 정보를 **버리는** 게 아니다. § 4.2 가 안 쓸 뿐, § 4.4 에서 100% 사용한다.

- G2 : "결과적으로 task 가 됐어?" → $s_T$ 만 보면 충분 (P1 + P2 덕)
- CPD : "그래서 어디서 위태로웠어?" → 매 step 의 $r_t$ 봐야 함

이 깔끔한 분리가 **Theorem 1 (G2 ↔ G1 consistency) 의 증명을 가능하게 함** —
G2 가 G1 과 일치해야 하는 건 "1-bit 라벨" 위에서이고, 그 라벨은 endpoint 의
함수이므로, A1 (Lipschitz φ) + A2 (ρ-cover) + A3 (oracle stability) 만으로
consistency 가 따라온다.

---

## 7. 실증 (우리 데이터의 4 가지 케이스)

`reports/exp2/critical_phase/per_episode.json` 기준:

| 케이스 | $\varphi(s_T)$ 가 $\varepsilon$-ball 안? | G2 라벨 | GT | 일치? |
|---|---|---|---|---|
| ep115 (정상 성공)         | yes (demo end 근처에서 멈춤) | succ | succ | ✓ |
| ep118 (초반 실패, 표류)   | no  (중간 어디서 멈춤)        | fail | fail | ✓ |
| **ep149 (어수선 → 성공)** | **yes** (결국 도착)           | **succ** | succ | ✓ |
| ep068 (장기간 표류)       | no                            | fail | fail | ✓ |

**ep149 가 핵심 케이스**: critical fraction 63%, longest_run = 281 step.
경로 기반 라벨러였으면 무조건 fail 로 찍었을 것. 그러나 $s_T$ 만 보는 G2 는
**올바르게 success** 로 라벨링, 이후 CPD scoring 이 "그런데 280 step 동안
위태로웠다" 를 따로 잡아냄. 두 모듈이 각자 할 일을 하는 것.

---

## 8. 한계 (paper 에 솔직히 써야 할 것)

1. **"성공으로 끝났지만 안전하지 않게 끝난" 케이스 못 잡음**
   - 예: 물건 부서뜨리고 결과만 성공.
   - 시뮬에선 거의 안 나오지만 실제 로봇에선 약점.
2. **시간 제한 ($T_\text{max}$) 의존**
   - 더 오래 줬으면 성공했을 rollout 이 timeout 시 fail 로 라벨링.
   - G2 가 아니라 환경 설정의 문제이긴 함.
3. **Multi-modal goal 에서 부서짐**
   - 여러 정답 endpoint 가 있는 task → $g$ = mean 이 깨짐.
   - § 4.2 가 암묵적으로 "single goal cluster" 를 가정.
4. **End cluster 가 좁다는 게 실증적 관찰**
   - § 2 의 메커니즘은 "왜 일어나는지" 의 직관 설명이고, 수학적 증명은 아님.
   - LIBERO 외 task 에서 P1 이 깨질 가능성 있음.
   - paper appendix 에 P1 sanity figure 필요: $\{\|\varphi(s_T^{(i)}) - g\|\}_i$ 의
     histogram + $\{\|\varphi(s_t^{(i)}) - g\|\}_{t<T,i}$ 의 histogram 비교.

---

## 9. 한 문장 bridge (paper 본문 삽입용)

> § 4.1 의 contrastive 학습은 입력 proprio 의 demo 간 물리적 유사성과 함수성
> 제약으로 인해 같은 task 의 성공 endpoint 들을 latent space 한 영역에 자동으로
> 모아준다 (P1). § 4.2 의 G2 labeler 는 이 P1 을 그대로 활용하여 "$\varphi(s_T)$ 가
> endpoint cluster 안에 들어왔는가" 로 1-bit 성공 라벨을 매긴다. trajectory 전체의
> 시간 정보는 § 4.1 에서 latent 의 task-progress 좌표축으로 압축되어 마지막 점에
> 응축되어 있으므로, 라벨링 시점에 $s_T$ 한 점만 봐도 충분하다.

이 문장을 § 4.1 끝 또는 § 4.2 첫 줄에 삽입하면 절 간 점프가 사라진다.
