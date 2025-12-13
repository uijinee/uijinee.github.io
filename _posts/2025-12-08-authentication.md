---
title: "4. Authentication"
date: 2025-12-08 12:00:00 +0900
categories: ["Math", "Cryptography"]
tags: ["cryptography"]
use_math: true
---

# Entity Authentication

## 1. Identification

### 1) Attacks against Identification 
공격 종류
- Impersonation
- Replay
- Interleaving


### 2) Type of Identification

> #### Password based
>
> 일반적인 password는 entropy가 낮아 추측이 쉽다는 문제가 있다. 즉, 고정된 패스워드를 사용할 경우 ⅰ) 도청시, ⅱ) 패스워드 테이블이 유출될 경우 위험하다.<br>
> 이를 방지하기 위해 다음과 같은 방식들을 사용할 수 있다.
>
> - Hashed Password<br> 
>   : 사용자의 Password를 Hashing한 후 서버에 저장하는 방식 (해시함수의 단 방향성 이용)<br>
>   $\rightarrow$ 사용자들이 많이 사용하는 password나 복잡하지 않은 암호의 경우 공격자가 쉽게 유추할 수 있음(dictionary attack)
>
> - Salted Password<br>
>   : Password + Salt를 Hashing한 후 서버에 저장하는 방식 (Salt는 사용자 별 고유의 Salt를 갖도록 함)<br>
>   $\rightarrow$ Salt는 비밀 값이 아니기 때문에, 공격이 불가능해지는 것이 아니라 여전히 비싸게 만드는 방식이다.(dictionary attack 가능)
> 
> - One-Time Password(OTP)<br>
>   : 한 번 쓰고 버리는 비밀번호(매번 다른 숫자가 나오고, 그 순간에만 유효)
>
>   - Synchronized OTP<br>
>     : 사용자와 서버가 Seed를 공유한 후, 같은 기준(시간, 이벤트)으로 비밀번호를 생성
>
>   - Non-Synchronized OTP<br>
>     : 사용자가 hash chain의 값(`w5`) 1개 서버로 전송 후 저장, 이후 사용자는 `w4`를 보내고 서버는 `w5=H(w4)`인지 확인 후, `w4`저장
> 
> ---
> #### Symmetric Key based
> 
> - Timestamp 기반<br>
>   ⅰ) A가 B에게 현재 시간 $t_A \Vert B$ 를 암호화 하여 전송<br>
>   ⅱ) B는 공유키 K로 이를 복호화 하여 A를 인증
> 
> - 난수 기반<br>
>   ⅰ) B가 A에게 난수 $r_B$를 전달<br>
>   ⅱ) A는 $r_B \Vert B$를 암호화하여 다시 전달<br>
>   ⅲ) B는 이를 복호화 하여 r_B가 맞는지 확인 후 A를 인증
> 
> - 난수 기반(상호인증)<br>
>   ⅰ) B가 A에게 난수 $r_B$를 전달
>   ⅱ) A도 난수 $r_A$를 생성한 후 $r_A \Vert r_B \Vert B$를 암호화 하여 전달<br>
>   ⅲ) B가 이를 복호화 하여 A를 인증한 후에, 다시 A에게 $r_A \Vert r_B$를 암호화하여 전달<br>
>   ⅳ) A는 이를 복호화하여 B를 인증
> 
> ※ 여기에 추가적으로 MAC을 통해 무결성을 보장할 수 있음
> 
> ---
> #### Public Key based
>
>
>
> vv

---
## 2. Biometric Recognition

### 1) 구성 요소

![alt text](/assets/img/post/cryptography/biometricsystem_component.png)

> - Sensor: 지문 스캐너, 얼굴 카메라, 마이크 등 신호 채집 장치
> - Feature Extractor: 센서의 원본 데이터를 정제하여 feature만 뽑아냄
> - Template Database: 사용자들의 생체 Feature template 저장
> - Matcher: 입력된 특징과 유사도 점수 계산
>
> ---
> #### 인증방식
>
> - Verification(1:1)<br>
>   : DB에 저장된 x의 템플릿과만 비교하여 matching score가 threshold보다 높으면 인증
> 
> - Identification(1:n)<br>
>   : DB에 있는 모든 사용자와 비교해 가장 점수가 높은 사람이 threshold보다 높을 경우 인증

### 2) 특징

- (Uniqueness) Inter-user similarity가 작아야 함<br>
: False Match를 줄일 수 있음
- (Permanence) Intra-user variation이 작아야 함<br>
: False non-match를 줄일 수 있음

> #### FAR & FRR
> 
> | | |
> | $$I_1: (x_{11}, x_{12}, ..., x_{1t}) \\ I_2: (x_{21}, x_{22}, ..., x_{2t}) \\ ... \\ I_n: (x_{n1}, x_{n2}, ..., x_{nt}) $$ | ![alt text](/assets/img/post/cryptography/frr_far.png)<br> FAR과 FRR의 trade off 관계를 고려해 적절한 threshold를<br> 설정 해야한다. |
>
> - genuine score: 같은 사람의 두 샘플을 비교하는 경우의 수<br>
>   : ${n \choose 2} \times n$
> 
> - Impostor score: 서로 다른 사람의 샘플끼리 비교하는 경우의 수<br>
>   : $t^2 \times {n \choose 2}$
>
> - FAR(False Accept Rate): 남을 나라고 잘못 인증하는 경우<br>
>   : $\text{FAR}(\eta) = p(s \geq \eta \;\vert\; \text{impostor})$
> 
> - FRR(False Reject Rate) 나를 나라고 인증하지 못하는 경우<br>
>   : $\text{FRR}(\eta) = p(s \leq \eta \;\vert\; \text{impostor})$
>
> ---
> #### 공격 모델
>
> | 유형         | 특징                        | 문제 발생              |
> | ---------- | ------------------------- | ------------------ |
> | **Sheep**  | 이상적 사용자                   | 거의 문제 없음           |
> | **Goats**  | 본인 매칭이 불안정 (intra-user ↑) | **FRR ↑** (본인을 거절) |
> | **Lambs**  | 다른 사람과 비슷함 (inter-user ↑) | **FAR ↑** (타인을 허용) |
> | **Wolves** | 타인을 속일 수 있음               | 보안적 위협             |
>
> 
> ※ 직업군별로 sheep이 될 수 있는 biometric trait이 다르기 때문에 trait의 적절한 선택이 필요하다.

### 3) Fingerprint Recognition

지문은 크게 Ridge(선)와 Valley(골) 패턴으로 구성되어 있다.<br>
이러한 지문을 찍는 방식은 ⅰ) Rolled, ⅱ) Plain, ⅲ) Latent(잠재 지문) 등 으로 나뉘게 된다. 이때, 각각의 지문에 대해 정확하게 인식하도록 하는 것이 중요하다.

> #### Level
> 
> ![alt text](/assets/img/post/cryptography/fingerprint_lvl.png)
> 
> (왼쪽부터 (a) 원본, (b) level 1, (c) level 2, (d) level 3)
> 
> 지문은 수집된 해상도에 따라 level 3 feature 까지 추출할 수 있다.
> 
> ![alt text](/assets/img/post/cryptography/loop_delta.png)
> - **Level 1**: 다음 점들의 패턴을 통해 지문을 Plain arch, Tented arch, Left/right loop, Whorl, Twin loop로 분류할 때 사용
>   - (a) Loop: ridge가 돌아 나가는 형태
>   - (b) Delta: 세모 형태로 ridge가 모이는 점
>   - (c) Whorl: Loop 2개가 모여 ridge가 하나의 점으로 모이는 형태
>   - (d) Non-Singularity: ridge나 delta가 없을 때
>
> ![alt text](/assets/img/post/cryptography/ending_bifurcation.png)
> - **Level 2**: 다음 점들의 위치와 방향 통해 사람을 인증할 수 있음
>   - Ridge ending: Ridge가 끝나는 지점
>   - Ridge bifurcation: Ridge가 갈라지는 지점 
>
> ![alt text](/assets/img/post/cryptography/fingerprint_lvl3.png)
> - **Level 3**: 1000dpi 이상의 고해상도 장비를 통해 관찰 가능
>
> 보통 인증은 level 1, 2를 같이 활용하는 경우가 많고, 경우에 따라 level 3을 활용하기도 한다.
> 
> ---
> #### pipeline
>
> ![alt text](/assets/img/post/cryptography/fingerprint_pipeline.png)
> 
> ⅰ) Orientation & Frequency Estimation<br>
>  $\quad$: 부분 이미지에 대해 DFT(Discrete Fourier Transform)을 적용하여 방향과 freq 추출<br>
>  $\quad$: Smoothing 과정을 거처 방향 보정
>
> | singularity filter | Poincare index |
> |:---:|:---:|
> | ![alt text](/assets/img/post/cryptography/singularity_filter.png) | ![alt text](/assets/img/post/cryptography/poincare_index.png) |
> 
> ⅱ) Singularity Extraction<br>
>  $\quad$: **(Point)** Poincare index등을 활용하여 singularity를 판단한다.(0: non, $\frac{1}{2}$: loop, $-\frac{1}{2}$ delta, 1: whorl) <br>
>  $\quad$: **(Direction)** Ridge의 방향과 Reference(loop delta)의 방향을 이용해 loop와 delta의 방향을 설정(e.g. $D_{loop} = \tan^{-1} (\frac{\sum_W \sin2(O(x, y) - RO(x, y))}{\sum_W \cos2(O(x, y) - RO(x, y))})$)
>
> ![alt text](/assets/img/post/cryptography/ridge_extraction_pipeline.png) | ![alt text](/assets/img/post/cryptography/gabor_example.png) | ![alt text](/assets/img/post/cryptography/bi_thining.png)
> 
> ⅲ) Ridge Extraction<br>
>   $\quad$ **(Gabor Filtering)**: 다양한 방향의 필터를 사용하여 융선을 선명하게 만든다.<br>
>   $\quad$ **(Binarization)**: 필터링된 이미지를 흑백(0과 1)으로 변환한다.<br>
>   $\quad$ **(Merge)**: 이진화된 이미지들을 하나로 합친다.<br>
>   $\quad$ **(Thining)**: 융선의 두께를 1픽셀로 얇게 만들어 뼈대만 남긴다.
>
> ![alt text](/assets/img/post/cryptography/minutiae_detection.png) | ![alt text](/assets/img/post/cryptography/minutiae_direction.png) | ![alt text](/assets/img/post/cryptography/minutiae_remove.png)
> 
> ⅳ) Minutiae Extraction<br>
>   $\quad$ **(Minutiae Detection)**: bifurcation과 ending을 찾아 표시한다.<br>
>   $\quad$ **(Calculate Direction)**: bifurcation과 ending의 방향을 계산한다.<br>
>   $\quad$ **(Remove)**: 테두리부분같은 spurious minutiae를 제거한다.<br>
> 
> ![alt text](/assets/img/post/cryptography/minutiae_matching.png)
> 
> ⅴ) Matching<br>
>   $\quad$ **(Alignment)**: bifurcation과 ending을 찾아 표시한다.<br>
>   $\quad$ **(Pairing)**: bifurcation과 ending의 방향을 계산한다.<br>
>   $\quad$ **(Scoring)**: 테두리부분같은 spurious minutiae를 제거한다.<br>