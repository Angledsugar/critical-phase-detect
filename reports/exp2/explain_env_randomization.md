# 환경 랜덤화 (object / goal 위치 변동) 가 TLDR + G2 에 미치는 영향

> 같은 task / 같은 prompt 라도 LIBERO 는 episode 마다 scene 이 다름:
> - object 의 시작 위치가 다름 (init_state cycling)
> - (task 에 따라) goal 위치도 다를 수 있음
>
> 이 상황에서 § 4.2 G2 의 "성공 endpoint 가 cluster 된다" 는 가정이 어떻게
> 지켜지는지 (또는 부분적으로 깨지는지) 를 정리.

---

## 1. 의문 정리

§ 4.2 의 G2 labeler 는 다음 가정에 의존:

> "성공 demo 의 $\varphi(s_T)$ 들이 latent space 의 좁은 한 영역에 모인다 (P1)."

그런데 LIBERO 에서는:

```
   episode 1 :  can at (0.3, 0.2),  goal at (0.6, 0.1)
   episode 2 :  can at (0.4, 0.1),  goal at (0.6, 0.1)   ← can 만 다름
   episode 3 :  can at (0.3, 0.2),  goal at (0.5, 0.3)   ← goal 만 다름
   episode 4 :  can at (0.2, 0.3),  goal at (0.7, 0.0)   ← 둘 다 다름
```

성공한 rollout 의 마지막 robot pose 도 episode 마다 다를 텐데, $\varphi(s_T)$ 가
어떻게 한 영역에 모이는가?

---

## 2. 먼저, "Proprio 가 담는 것 vs 안 담는 것"

State $s_t \in \mathbb{R}^8$ 의 8 차원은 **로봇 자체의 kinematics 만**:

```
   s_t  =  [ ee_x, ee_y, ee_z,           ← end-effector 위치  (3)
             ee_rx, ee_ry, ee_rz,         ← end-effector 자세  (3)
             gripper_l, gripper_r ]       ← 그리퍼 개폐         (2)
```

**proprio 가 보는 것**:
- 로봇 손이 어디에 있는가
- 그리퍼가 얼마나 열려 있는가

**proprio 가 못 보는 것**:
- can 이 어디에 있는가
- goal 이 어디에 있는가
- 다른 물체들의 위치

→ 핵심 함의: **proprio 는 "world state" 가 아니라 "robot self-state" 다**. 이 비대칭이
G2 가 환경 랜덤화에 robust 한 근거 (그리고 §4.2 limitation 의 근거이기도 함).

---

## 3. Trajectory 의 variability profile

LIBERO-Long task00 에서 episode 마다 어디가 변하는지:

| 단계 | 로봇 proprio | 왜 |
|---|---|---|
| **t = 0 (시작)** | 항상 동일 (home pose) | 모든 episode 가 같은 home pose 에서 시작 |
| **t ∈ (0, T) (중간)** | **episode 마다 크게 다름** | 로봇이 다양한 위치의 object 를 잡으러 감 |
| **t = T (끝)** | episode 마다 약간 다름 (goal 위치에 의존) | 로봇이 object 를 goal 에 놓음 |

ASCII 로 표현하면:

```
                     proprio 의 episode 간 분산
                     │
                     │                      ╱─────╲
                     │                ╱─────       ╲─────╲
                     │          ╱─────                     ╲────╲
                     │     ╱────                                  ╲────
                     │   ──                                            ────
                     │ ──                                                  ──
                     └──────────────────────────────────────────────────────────→  t
                       t=0                                                   t=T
                     (home,                  (object reach,                (goal,
                      모든 ep 동일)            episode 마다 다름)            goal 좌표에 의존)
```

→ **시작은 좁고, 중간은 크게 벌어지고, 끝은 다시 좁아지는 "모래시계" 모양**.

### 3.1 왜 끝이 다시 좁아지는가 — case 별

**Case A : object 위치만 랜덤 (goal 고정)**
- 모든 demo 가 같은 goal 위치에 object 를 놓음.
- 끝 robot proprio ≈ "goal 위 hover + gripper 열림" → episode 무관하게 거의 동일.
- → **end cluster 가 매우 좁음** (LIBERO 의 일반적인 case).

**Case B : object 와 goal 모두 랜덤**
- demo 마다 끝 robot proprio 가 goal 좌표에 따라 다름.
- 그러나 randomization 범위가 한정적 (workspace 일부) → endpoint 들이 한 region 에
  분포 (이상값은 95% quantile $\varepsilon$ 으로 컷).
- → **end cluster 가 case A 보다 넓지만 여전히 한정된 region**.

**Case C : 진짜 unconstrained goal (이론적 worst case)**
- goal 이 전체 workspace 에 균일 분포.
- endpoint 가 workspace 전체에 퍼짐.
- → P1 깨짐. G2 가 failure rollout 도 ε-ball 안에 포함시킴 (false positive 폭증).
- → LIBERO 에서는 이 case 가 거의 안 나옴.

---

## 4. 그러므로 latent space 도 모래시계 모양이 됨

§ 2 의 proprio 분산 profile 을 latent 로 전달:

```
                       latent space (한 방향으로 simplification)

      start cluster                                      end cluster
      (모두 home →                                       (goal 좌표에
       latent 한 점 부근)                                 의존하는 좁은 region)
            ●                                                  ●
            ●●                                              ●●●
           ●●●●                                         ●●●●●●
          ●●●●●●●            ╲                        ●●●●●●●●●
         ●●●●●●●●●            ╲       ╱             ●●●●●●●●●●●●
        ●●●●●●●●●●●●           ╲     ╱             ●●●●●●●●●●●●●
       ●●●●●●●●●●●●●           ╲   ╱             ●●●●●●●●●●●●●●●
      ●●●●●●●●●●●●●●           ╲ ╱             ●●●●●●●●●●●●●●●●
     ●●●●●●●●●●●●●●●            ●            ●●●●●●●●●●●●●●●●●●
                          (중간 영역,                         ↑
                           episode 마다              G2 의 g 와
                           다양한 path                ε-ball 이 잡는 영역
                           가 통과)
```

이 모양이 G2 의 작동을 가능하게 함:
- $g = \mathrm{mean}_i \varphi(s_T^{(i)})$ → end region 의 중심
- $\varepsilon = \mathrm{Quantile}_{0.95}\big(\{\|\varphi(s_T^{(i)}) - g\|\}_i\big)$ →
  end region 의 95% 반경 (Case B 의 goal randomization 만큼 자동으로 늘어남)
- $\hat{y}(\tau) = \mathbb{1}[\|\varphi(s_T) - g\| \leq \varepsilon]$ →
  end region 안에 들어왔는가

→ **랜덤화로 인한 endpoint 분산은 $\varepsilon$ 으로 흡수**. 사람이 threshold 를
조정할 필요 없이 demo 통계로 자동 calibration.

---

## 5. 왜 시작 cluster 가 좁은 게 중요한가 (가끔 간과되는 포인트)

§ 4.1 의 triplet loss 가 작동하려면 demo 간 비교가 의미를 가져야 함.

만약 시작 proprio 도 episode 마다 크게 다르면:

```
   demo i :  start (random A) → ... → end (random)
   demo j :  start (random B) → ... → end (random)
```

이 경우 $\varphi(s_0^{(i)})$ 과 $\varphi(s_0^{(j)})$ 가 latent 의 서로 다른 점.
"task 진행도 0%" 라는 공통 좌표가 정의 안 됨 → start cluster 깨짐 → §4.1 의
"매끄러운 path" 가 demo 별로 따로 놀음 → endpoint 도 따로 놀게 됨.

LIBERO 는 다행히 **home pose 가 고정**이라 시작 cluster 가 항상 좁음. 이게 TLDR +
G2 의 **암묵적 전제** 중 하나.

---

## 6. 실증 — 우리 task00 200 episode 통계

`reports/exp2/critical_phase/per_episode.json` 기준 (성공 187 ep):

```
   ε = 95% quantile of  ||φ(s_T) − g||  (LOO 가 아닌 demo 기준)
     ≈  특정 작은 값 (paper § 5.1 의 Table 1)

   성공 rollout 의 ||φ(s_T) − g||  분포 :
       대부분 ε 이내, 95% 가 ε 안쪽 (정의상)

   실패 rollout 의 ||φ(s_T) − g||  분포 :
       대부분 ε 의 수 배 ~ 수십 배 (latent 의 "중간 어디" 에 멈춤)
```

→ **end cluster 의 분산 < failure 의 분산 (수 배 ~ 수십 배 차이)**.
이게 G2 가 200 ep 에서 GT 와 100% 일치하는 이유 (`reports/exp1/summary.md` Exp1).

만약 환경 랜덤화가 너무 심해서 end cluster 분산이 failure 분산만큼 커진다면 G2 는
무너짐. LIBERO-Long task00 은 그 임계점에서 멀리 떨어져 있다는 게 실험적 결론.

---

## 7. 어떤 랜덤화 시나리오에서 G2 가 깨지는가

P1 (endpoint clustering) 이 깨지는 조건:

```
   end cluster 의 분산  >  failure rollout 들의 분산
```

이게 일어나는 경우:

1. **Goal 이 전체 workspace 에 균일 분포**
   - 성공 endpoint 가 workspace 전체에 산포 → 분산 큼
   - failure 도 workspace 일부에 산포 → 분산 비슷
   - → G2 false positive 폭증
   - → 대응: per-goal-region 별 $g, \varepsilon$ (goal-conditioned G2) 필요
   - → 본 paper § 5.6 future work 후보

2. **여러 task 가 한 buffer 에 섞임**
   - task 마다 다른 end cluster → 합치면 multimodal
   - mean 으로 정의된 $g$ 는 mode 사이 어디 — 어느 cluster 도 잘 안 잡음
   - → 대응: per-task $g, \varepsilon$ (현재 우리가 하는 방식)

3. **Multi-modal success** (한 task 에 여러 정답)
   - "object 를 책상 어느 곳에든 놓으면 성공" 같은 task
   - end cluster 가 여러 mode → 위와 같은 문제
   - → 대응: KDE 기반 $\tilde{f}_+$ 사용 (mean·radius 가 아니라 density)
   - → 본 paper § 4.4 의 CPD score 는 이미 KDE 라서 multi-modal 대응 가능,
     그러나 § 4.2 G2 의 라벨링은 단일 cluster 가정.
   - → § 5.6 future work : "$g, \varepsilon$ 를 KDE 로 확장" 필요.

4. **Goal 이 proprio 에 안 보임**
   - 예: "박스 안에 물건 넣기" → 박스 위치가 proprio 에 안 잡힘
   - 같은 robot proprio 가 두 goal 에서 발생 가능 → identifiability 깨짐
   - → 본 paper 의 scope 밖 (proprio + LIBERO 의 reachable goal 가정)

---

## 8. Theorem 1 의 가정 (A1, A2, A3) 과 랜덤화

paper § 4.7 의 Theorem 1 (G2 ↔ G1 consistency) 의 가정이 어떻게 영향받는가:

| 가정 | 내용 | 환경 랜덤화의 영향 |
|---|---|---|
| **A1 (Lipschitz φ)** | $\varphi$ 가 Lipschitz | 랜덤화와 무관. encoder 자체의 regularity. |
| **A2 (ρ-cover)** | buffer 가 success / failure manifold 를 $\rho$-cover | **랜덤화 클수록 manifold 가 커짐 → 더 많은 demo 필요** |
| **A3 (oracle stability)** | G1 (oracle predicate) 이 작은 perturbation 에 stable | LIBERO 의 goal predicate ("object on plate") 는 plate 위치의 작은 perturbation 에 stable → OK |

→ Theorem 1 자체는 **demo 수가 충분하면** 깨지지 않음. 다만 랜덤화가 클수록 필요한
demo 수가 커짐. 이게 § 5.4 의 "F1 ceiling 은 신호가 아니라 failure pool" 분석과
연결: 200 ep 으로 task00 의 endpoint 분산을 cover 하기엔 충분, 그러나 더 unconstrained
한 task 에서는 demo 가 더 필요할 수 있음.

---

## 9. 직관 — 왜 "robot-egocentric proprio" 가 의외로 robust 한가

LIBERO 같은 manipulation task 의 구조적 특징:

```
   "성공한다" ⟺ "object 를 goal config 에 놓는다"
              ⟺ "로봇이 그 object 를 goal 위로 가져가서 놓는다"
              ⟺ "로봇 end-effector 의 마지막 위치 ≈ goal 위치"
```

→ **로봇의 마지막 proprio 가 goal 좌표를 implicit 하게 인코딩**.

object 시작 위치가 어디든 (case A), goal 이 어디든 (case B), 마지막 로봇 pose 는
**goal 좌표의 함수**. proprio 가 world state 를 못 보지만, **성공의 정의가
"로봇이 goal 에 도달했는가" 이므로** 로봇 proprio 가 결국 그 정보를 담음.

이건 manipulation task 의 좋은 특징이지, 일반 RL setting 에서 항상 성립하는 건 아님.
예를 들어:
- "어디서든 좋으니 1분 동안 살아있어라" → 시간 기반 task → proprio 와 success
  무관 → G2 작동 안 함.
- "방을 청소해라" → object 분포 자체가 success → proprio (로봇만) 으로 안 됨.

LIBERO 처럼 **goal-config-based manipulation** 이 우리의 sweet spot.

---

## 10. 한 줄 요약

> 환경 랜덤화는 trajectory 의 **중간** 을 흩어 놓지만, **시작 (home) 과 끝 (goal)**
> 은 좁은 region 에 모인다 (모래시계 모양). 로봇의 proprio 가 robot-egocentric 이라
> world state 를 못 보지만, manipulation task 의 정의상 마지막 robot pose 가 goal
> 좌표를 implicit 하게 인코딩하므로, **성공 endpoint 의 latent 분산이 failure 의
> 분산보다 작게 유지된다**. § 4.2 의 $\varepsilon$ quantile 이 이 분산을 자동으로
> 흡수하므로 사람이 threshold 를 조정할 필요 없음.
> Goal 이 unconstrained 하거나 multi-modal 인 task 에서는 이 가정이 깨질 수 있으며
> § 5.6 의 goal-conditioned G2 / KDE-based G2 가 future work.

---

## 11. paper 본문에 추가해야 할 한 줄 (§ 4.2 또는 § 5.1)

> "LIBERO 의 init-state randomization 은 trajectory 의 중간 부분만 흩어 놓으며,
> 로봇이 home pose 에서 시작해 goal config 에 도달하는 구조상 마지막 robot proprio
> 는 좁은 region 에 모인다. $\varepsilon$ 의 95%-quantile 정의가 이 자연스러운 분산을
> 자동으로 calibration 하므로, 환경 랜덤화 자체가 G2 의 정확도를 떨어뜨리지 않는다
> (§ 5.1 Table 1, Exp1 에서 G2 가 GT 와 100% 일치)."
