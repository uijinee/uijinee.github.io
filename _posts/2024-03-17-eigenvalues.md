---
title: "3. Eigenvalues"
date: 2024-03-17 22:00:00 +0900
categories: ["Math", "Linear Algebra"]
tags: ["math", "eigenvalue", "eigenvector", "eigendecomposition"]
use_math: true
---

# 

## 1. 고윳값 분해(Eigen Value Decomposition)

### 1) 고윳값, 고유벡터
$A\mathbb{x} = \lambda \mathbb{x}$<br>

&#8251; $A$: 행렬, $\mathbb{x}$: 0이 아닌 벡터, $\lambda$: 스칼라

- $\mathbb{x}$ = 고유 벡터<br>
- $\lambda$ = 고유값

> 0이 아닌 어떤 벡터 $\mathbb{x}$에 대해 행렬 A에 의한 선형 변환의 결과가 자기 자신의 상수배가 될 때(즉, 방향이 변하지 않을 때) 상수값과 벡터를 말한다.
>
> ---
> #### 고유값
> 
> $(\lambda - A) \mathbb{x} = 0$이 $\mathbb{x} \neq 0$인 해(자명하지 않은 해)를 가져야 한다. <br> 
> 만약 $(A-\lambda I)$의 역행렬이 존재한다고 하면 $\mathbb{x}=0$만을 해로 가지기 때문에 처음 가정과 모순이다.
> 
>> 따라서 특성방정식 $det(\lambda I - A) = 0$ 을 만족하는 &lambda;가 고유값이다.
>
> 또, 이 고윳값으로 대각합과 행렬식을 구할 수 있다,
> - $det(A) = \prod \limits_{i=1}^N \lambda_i$
> - $tr(A) = \sum \limits_{i=1}^N \lambda_i$
>
> ---
> #### 고유벡터
> 
>> 고유값 &lambda; 를 먼저 구한 후,<br>
>> $(\lambda - A) \mathbb{x} = 0$에 대입하여 $\mathbb{x}$를 구한다.


### 2) 대각화

> 어떤 행렬 $A$의 고유값과 고유벡터를 알아내면 대각화를 쉽게 할 수 있다.<br>
> 즉, 이로써 계산상의 이점을 얻을 수 있다.
>
> ---
> **대각화가능성 및 대각화**
>
> 고유값과 고유벡터로 부터 행렬얼 어떻게 대각화 할 수 있는지 알아보자.
>
> - ⅰ. 행렬 A의 고유값과 고유벡터가 각각 $(\lambda_1, \lambda_2, ... , \lambda_n), \quad (\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_n)$ 이라고 하자.
> 
> - ⅱ. 고유값과 고유벡터의 정의에 의해 다음과 같이 쓸 수 있다.<br>
> $$
> \begin{pmatrix}
>   \mathbf{Av}_1 = \lambda_1 \mathbf{v}_1\\ 
>   \mathbf{Av}_2 = \lambda_2 \mathbf{v}_2\\ 
>   \vdots \\ 
>   \mathbf{Av}_n = \lambda_n \mathbf{v}_n 
> \end{pmatrix}
> \rightarrow
> \mathbf{A}
> \begin{pmatrix}
>   \mathbf{v}_1\\ 
>   \mathbf{v}_2\\ 
>   \vdots \\ 
>   \mathbf{v}_n
> \end{pmatrix}^T
> =
> \begin{pmatrix}
>   \mathbf{v}_1\\ 
>   \mathbf{v}_2\\ 
>   \vdots \\ 
>   \mathbf{v}_n
> \end{pmatrix}^T
> \begin{pmatrix}
>   \lambda_1 & 0 & ... & 0 \\ 
>   0 & \lambda_2 & ... & 0 \\ 
>   \vdots & \vdots & \ddots & \vdots \\ 
>   0 & 0 & ... & \lambda_n
> \end{pmatrix}
> $$<br>
> *($\mathbf{v}_i$는 열벡터임을 주의)*
>
> - ⅲ. 이때, $P = \begin{pmatrix} \mathbf{v}_1 & \mathbf{v}_2 & ... & \mathbf{v}_n \end{pmatrix}$ 의 역행렬이 존재하기 위해서는 원소들이 <u>선형 독립</u>이어야 한다.
> 
> - ⅳ. 만약 $P$의 역행렬이 존재한다고 하면 다음과 같이 쓸 수 있다.
> $$
> \mathbf{A} \mathbf{P} = \mathbf{P} \mathbf{\Lambda} \rightarrow \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^{-1} \\
> \&\quad \mathbf{P} = \begin{pmatrix} \mathbf{v}_1 & \mathbf{v}_2 & ... & \mathbf{v}_n \end{pmatrix} \\
> \&\quad \mathbf{\Lambda} = 
> \begin{pmatrix}
>   \lambda_1 & 0 & ... & 0 \\ 
>   0 & \lambda_2 & ... & 0 \\ 
>   \vdots & \vdots & \ddots & \vdots \\ 
>   0 & 0 & ... & \lambda_n
> \end{pmatrix}
> $$
>
> ---
> **활용**
> 
> - 이 대각화의 결과($A = P^{-1} D P$)는 닮음과도 관련이 있다.<br>
>   $\rightarrow$ 닮음: $AP = PB$, 즉 두 정사각행렬이 같은 선형 변환의 서로 다른 기저에 대한 표현임을 나타내는 관계
>  
> - 행렬 A를 대각화하면 다음과 같은 식을 활용할 수도 있다.<br>
> $A^n = (P^{-1} \Lambda P) ... (P^{-1} \Lambda P) = (P^{-1} \Lambda^n P)$

### 3) 직교대각화

> - 직교대각화란 고유벡터($\mathbf{U} = \begin{bmatrix}\mathbf{u}_1 & \mathbf{u}_2 & ... & \mathbf{u}_n \end{bmatrix}$)들이 모두 직교 한다는 뜻이다.<br>
>   ⅰ. $(if \; i\neq j) \qquad \mathbf{u}_i^T\mathbf{u}_j = 0$<br>
>   ⅱ. $(if \; i=j) \qquad \mathbf{u}_i^T\mathbf{u}_j = 1$
>
> - 대칭행렬($A^T=A$)을 대각화하면 직교대각화가 된다.<br>
>  $\rightarrow$ <u>대칭행렬의 대각화와 직교대각화는 동치이다.</u>
>
> ---
> **대칭행렬의 성질**
>
> 1. 고유값이 항상 실수이다.<br>
> 2. 고유벡터는 모두 직교한다.<br>
>   $\rightarrow \mathbf{U}^T\mathbf{U} = \mathbf{U}\mathbf{U}^T = I$<br>
>   $\rightarrow \mathbf{U}^{-1} = \mathbf{U}^T $
> 
> 3. PSD (Positive Semi Definite)이다<br>
>   $\rightarrow$ 모든 고유값 $\lambda_i \geq 0$
>
> ---
> **직교대각화**
>
>> $A$가 대칭행렬이라면, 대각화시 직교대각화를 할 수 있고,<br>
>> $A = \mathbf{U \Lambda U}^T$ 이다.
> 
> - Rank-1 matrix 분해<br>
> $$
> A = \mathbf{U \Lambda U}^T = \begin{pmatrix}u_1 & u_2 & ... & u_n\end{pmatrix}
> \begin{pmatrix}
>   \lambda_1 & 0 & ... & 0 \\ 
>   0 & \lambda_2 & ... & 0 \\ 
>   \vdots & \vdots & \ddots & \vdots \\ 
>   0 & 0 & ... & \lambda_n
> \end{pmatrix} 
> \begin{pmatrix} u_1^T \\ u_2^T \\ \vdots \\ u_N^T \end{pmatrix} \\
> = \begin{pmatrix} \lambda_1 u_1 & \lambda_2 u_2 ... \lambda_n u_n\end{pmatrix}
> \begin{pmatrix} u_1^T \\ u_2^T \\ \vdots \\ u_N^T \end{pmatrix} \\
> = \sum \limits_{i=1}^n \lambda_i u_i u_i^T
> $$
>
> - 즉, $\lambda_i$ 는 단위벡터 $u_i$ 방향에 대한 Gain으로 생각할 수 있다.

---
## 2. 특이값 분해(Singular Value Decomposition)

사용분야: 알고리즘, 자료압축

### 1) 특이값

고유값은 정방행렬에서만 정의 되었기 때문에, 정방행렬이 아닌 $A_{M \times N}$행렬에서는 정방행렬과 같은 방법으로는 대각행렬을 찾을 수 없다.

> **$A^TA$의 성질**
> 
> ---
> #### 직교대각화
> 행렬 $A_{M \times N}$는 $A^TA$행렬로 변환시켜 정방행렬로 만들어 대각행렬을 찾을 수 있는데, 이 때의 대각행렬은 직교대각행렬이다.
>
>> $A^TA$는 직교대각화가 가능하다.<br>
>> 이때, $A^TA$의 고유값은 음이 아니다.
> 
> ---
> #### 특이값
>
> $A^TA$의 고유값을 $\lambda_1, \lambda_2, ..., \lambda_n$이라고 할 때,
>> $\sigma_1 = \sqrt{\lambda_1},$<br>
>> $\sigma_2 = \sqrt{\lambda_2},$<br>
>> ...<br>
>> $\sigma_n = \sqrt{\lambda_n}$<br>
>> 을 행렬 $A$의 **특이값**이라고 한다.

### 2) 특이값 분해

대각화: $A = P^{-1} D P$ (단, A는 정방행렬)<br>
직교대각화: $A = U\sum V^T$ (단, A는 $M \times N$의 행렬)

> #### 대각화 
> 
> 대각화는 정방행렬 $A와$ 닮음,<br>
> 즉 $A= P^{-1} D P$를 만족하는 대각행렬을 찾는 문제였다.
>
> - $D$<br>
> : $A$의 고유값들로 이루어진 대각행렬
> 
> - $P$<br>
> : $D$의 고유값에 대응하는 열벡터 $\vec{p}$들로 이루어진 행렬
>
> ---
> #### 직교 대각화
> 
> 직교대각화는 $A$와 직교적으로 닮음,<br>
> 즉, $A = P^{T} D P$를 만족하는 대각행렬을 찾는 문제이다.
>
> 이때, 위 식은 $A = U\sum V^T$로 분해할 수 있다.
>
> - $\sum$<br>
> : $A$의 특이값이 주대각에 있고 나머지 원소가 0인 $M \times N$행렬<br>
> : $\sigma_1 > \sigma_2 > ... > \sigma_n$을 만족하도록 배열되어 있음
> 
> - $V^T$<bR>
> : $\sum$의 각 특이값에 대응하는 고유열벡터 $V=\{\vec{v_1}, \vec{v_2}, ..., \vec{v_n}\}$의 전치행렬
>
> &#8251; $AV = US$
>
> ---
> #### Rank-r 근사
>
> 행렬 A에 대해 특이값 분해를 수행하여 $A = U\sum V^T$를 얻었을 때, 열 행 전개를 이용하면
>
> $A = \sigma_1 \vec{u_1} {v_1}^T + \sigma_2 \vec{u_2} {v_2}^T + ... + \sigma_n \vec{u_n} {v_n}^T$로 표현 가능하다.
>
> 이때, $\sigma_1 > \sigma_2 > ... > \sigma_n$이므로 r번째 이하의 &sigma;를 0으로 취급하여 행렬을 압축하는 것을 Rank-r근사라고 한다.
>
>> 이때, Rank-r근사에 의한 최소제곱오차는<br>
>> $\underset{rank{\hat{A}} \leq{r}}{min} ||A- \hat{A}||_F = \sqrt{\sigma_{r+1}^2 + \sigma_{r+2}^2 + ... + \sigma_{m}^2}$이다.



---

## 1. LU-분해

사용분야: 컴퓨터 알고리즘

> **이유**
>
> 연립 1차 방정식을 풀 때, **가우스 소거법(행사다리꼴 이용)** 이나 **가우스-요르단 소거법(기약 행 사다리꼴 이용)** 을 사용하였다.<br>
> 하지만 이 방법은 다음과 같은 특징을 갖는 데이터에서 문제가 있다.
>
> - 컴퓨터 반올림 오차
> - 메모리 사용
> - 속도
>
> 따라서 위와 같은 문제점을 중요하게 여겨야 하는 큰 규모의 문제들을 푸는 데는 적합하지 않다.

---
## 2. 거듭제곱법

사용분야: 검색엔진

> **이유**
>
> 정방행렬의 고유값을 구하기 위해서는 특성방정식을 풀어야 한다.<br>
> 하지만 이 과정은 계산상 복잡하고 어렵기 때문에 대부분의 응용에서 직접 이용하기 힘들다.

---

### 4) 활용
> #### LU분해
> #### QR분해
> #### 그람-슈미트 과정
