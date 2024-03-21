---
title: "1. Convex Set"
date: 2024-03-14 22:00:00 +0900
categories: ["Math", "Convex Optimization"]
tags: ["math"]
use_math: true
---

# Convex Optimization

## 1. Optimization

### 1) 정의

![alt text](/assets/img/post/convexset/convexoptimization.png)

> **Object Function**을 주어진 **Constraints**하에, **Minimize**하는 것.<br>
> *(이를 "**Programming**, 계획법"이라고도 함. (Coding을 의미하는 것이 아님))*
>
>
> - 주로 이때, Minimize하는 정의역의 값(변수 값)을 찾는 것이 목표
>
> ---
> **Maximize**
>
> "$f(x)$"를 Maximize하는 문제는 "$-f(x)$"를 Minimize하는 문제로 변환이 가능하다.<br>
> 즉, Maximize문제와 Minimize문제는 근본적으로 같은 문제이다.
> 
> ---
> **Example**
>
> $f_0(x) = x^2-2x+4$, $(0 \leq x \leq 5)$ 일 때,<br>
> $f_0(x)$를 최소화 하는 x 찾기

### 2) 특징

> **모든 문제**는 최적화 문제로 변환이 가능하다.
>
> 하지만 아래의 예시들을 보면 알 수 있듯이 **Constraints의 존재** 때문에 Optimization을 하는 것은 대부분 매우 어렵다.
>
> ---
> **ex1. Machine Learning**
>
> - **Object Function**<br>
>   : Minimize Loss Function
> - **Constraints**<br>
>   : 가중치의 수, Layer의 수, Input Dimension, Output Class의 수, ... 
> 
> ---
> **ex2. Resource Planning**
>
> - **Object Function**<br>
>   : Minimize Total Cost
> - **Constraints**<br>
>   : 재료의 비용, 운송 비용, 시간적 한계, ...
> 
> ---
> **ex3. Physics**
>
> - **Object Function**<br>
>   : Maximize Entropy
> - **Constraints**<br>
>   : 초기 온도, 에너지 보존법칙, ...

### 3) Convex Optimization

![alt text](/assets/img/post/convexset/convexoptimization_table.png)

> **What?**
>
> Convex Optimization은 무엇일까?<br>
> Optimization 상황에서 다음과 같은 상황에 해당하는 경우를 의미한다.
> - Optimization Function $\rightarrow$ Convex Function
> - Constraints $\rightarrow$ Convex Set
> 
> ---
> **Why?**
>
> 그렇다면 Convex Optimization을 공부하는 이유는 무엇일까?
> 
> 대부분의 최적화 문제들은 풀기 어렵다고 알려져 있지만,<br>
> 이 중 Convex Optimization 문제들은 정형화된 방법(Known Algorithm)으로 푸는 것이 가능하다.
>
> 즉, 만약 Convex Optimization 문제가 아니라고 하더라도 다음과 같이 풀 수 있다.
> - Convex Optimization문제가 되도록 변형한다.
> - 최적의 해는 아니더라도 좋은 해(차선의 해)를 찾는다.
> 
>> 다시말해, 이 문제가 풀 수 있는 문제인지를 알기 위해<br>
>> **<u>이 문제가 Convex Optimization문제인지 먼저 판단하는 것이 중요하다.</u>**
> 

---
## 2. Some Sets

Convex Set은 Convex Function의 정의역 역할을 한다.<br>
이 챕터에서는 Convex Set의 정의를 알아보기 위해 몇가지 집합들의 공간에 대해 살펴보자.

### 1) Subspace 

![alt text](/assets/img/post/convexset/subspace.png)

> ⅰ. **Plane**
> 
>> $\begin{Bmatrix}\theta_1 \textbf{x}_1 + \theta_2 \textbf{x}_2 \| \theta \in \mathbb{R} \end{Bmatrix}$
>
> 위와 같이 여러 벡터들의 Linear Combination의 형태를 갖는다.
>
> - 이 공간을 최소한의 수의 벡터로 표현할 때, 이 벡터들을 기저벡터라고 한다.
>
> ---
> ⅱ. **Subspace**
>
> 조건-
>
> ---
> *Linear Combination 이란?*
>   - $\begin{Bmatrix}\textbf{x} = \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n \| \theta \in \mathbb{R} \end{Bmatrix}$

### 2) Affine Sets

![alt text](/assets/img/post/convexset/affineset.png)

> ⅰ. **Lines**
> 
>> $\begin{Bmatrix}\theta \textbf{x}_1 + (1-\theta)\textbf{x}_2 \| \theta \in \mathbb{R} \end{Bmatrix}$
>
> 위와 같이 가중치의 합이 1이되는 점들의 Linear Combination은 하나의 Line의 형태를 갖는다.
>
> 1. 증명<br>
>       - $\textbf{x}_2 + \theta(\textbf{x}_1 - \textbf{x}_2)$로 변환 가능<br> &rarr; $\textbf{x}_2$ 를 지나고 $ (\textbf{x}_1 - \textbf{x}_2)$ 와 평행한 점들의 모임
> 
> ---
> ⅱ. **Affine Sets**
>
>> $\begin{Bmatrix}\theta \textbf{x}_1 + (1-\theta)\textbf{x}_2 \in \mathcal{A} \| (x_1, x_2) \in \mathcal{A}, \theta \in \mathbb{R} \end{Bmatrix}$
>
> 즉, $\mathcal{A}$ 의 원소 2개를 이어 만든 직선이 전부 $\mathcal{A}$ 에 속해야 한다.
> 
> 1. 예시
>       - $\textbf{x}$ 의 <u>범위가 주어지지 않은</u> 평면
>       - $\textbf{x}$ 의 <u>범위가 주어지지 않은</u> 직선
>       - $S = \begin{Bmatrix} \textbf{x} \| A\textbf{x}=\textbf{b} \end{Bmatrix}$
>
> 2. *예시 3 증명*
>   - $\textbf{x}_1, \textbf{x}_2 \in S$ 라고 할 때, $\theta \textbf{x}_1 + (1-\theta) \textbf{x}_2$ 에 대하여<br>
>     $A \begin{Bmatrix} \theta \textbf{x}_1 + (1-\theta) \textbf{x}_2 \end{Bmatrix} = \theta A \textbf{x}_1 + A\textbf{x}_2 -\theta A \textbf{x}_2 = \theta \textbf{b} + \textbf{b} - \theta \textbf{b} = \textbf{b}$ 이기 때문에,<br>
>     $\theta \textbf{x}_1 + (1-\theta) \textbf{x}_2$(직선 위의 모든 점) 또한 $\mathcal{A}$ 에 속한다.

### 3) Convex Set

![alt text](/assets/img/post/convexset/convexset.png)

> ⅰ. **Line Segment**
>
>> $\begin{Bmatrix}\theta \textbf{x}_1 + (1-\theta)\textbf{x}_2 \| \theta \in \mathbb{R},  0 \leq \theta \leq 1 \end{Bmatrix}$
>
> Lines의 정의에서 $\theta$ 의 범위를 위와 같이 0과 1 사이로 한정하면 선분이 된다.
>
> ---
> ⅱ. **Convex Set**
> 
>> $\begin{Bmatrix}\theta \textbf{x}_1 + (1-\theta)\textbf{x}_2 \in \mathcal{A} \| (x_1, x_2) \in \mathcal{A}, \theta \in \mathbb{R},  0 \leq \theta \leq 1 \end{Bmatrix}$
>
> 1. 예시
>      - 다면체(다각형)
>      - 원, 타원
> 
> ---
> *Convex Combination 이란?*
>
>   - $\begin{Bmatrix}\textbf{x} = \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n \| \theta \in \mathbb{R}, \theta \geq 0, \sum \theta_i = 1\end{Bmatrix}$
>
> 이때, Convex Combination은 확률과 비슷한 성질을 갖는다는 점을 기억해 두자.

---
## 3. Convex Set

### 1) Convex Hull

![alt text](/assets/img/post/convexset/convexhull.png)

>
>> $Conv(S) = \begin{Bmatrix}\theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n \| x_i \in \mathcal{S}, \theta_i \geq 0 , \sum \theta_i = 1\end{Bmatrix}$
>
> - 정의<br>
>   - 어떤 점들로 표현할 수 있는 모든 Convex Combination을 원소로 갖는 집합<br>
>   - 즉, 집합 S를 포함하는 가장 작은 Convex Set
> 

### 2) Convex Cone

![alt text](/assets/img/post/convexset/convexcone.png)

>> $\begin{Bmatrix}\theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n \| x_i \in \mathcal{S}, \theta_i \geq 0 \end{Bmatrix}$
> 
> - 정의<br>
>   - 어떤 점들로 표현할 수 있는 모든 Conic Combination을 원소로 갖는 집합
> - **주의**: <u>Convex Cone은 반드시 원점을 포함한다.</u>
>   - 특정 $\theta_i=0$ 인경우: 가장자리
>   - 모든 $\theta_i=0$ 인경우: 원점
> 
> ---
> *Conic Combination 이란?*
> 
> - $\begin{Bmatrix} \textbf{x} = \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n \in \mathcal{S} \| x_i \in \mathcal{S}, \theta_i \geq 0 \end{Bmatrix}$

### 3) Hyperplanes

![alt text](/assets/img/post/convexset/hyperplane.png)

>> $\begin{Bmatrix} x \|a^T \textbf{x} = b \end{Bmatrix}, a \neq 0, b=a^T \textbf{x}_0$
>
> - 정의
>   - 벡터 $a$ 에 수직인 면
> - **주의**: 두 벡터 $a, b$ 의 내적은 $a^Tb$로 표현할 수 있다.
>   - 일반적인 벡터는 $n \times 1$ 의 열벡터를 나타내기 때문
>   - 평면의 방정식: $a^T(\textbf{x}-x_0) = (a, b, c) \cdot (x_1 - x_0, x_2 - x_0, x_3 - x_0)$
>

### 4) Halfspaces

![alt text](/assets/img/post/convexset/halfspace.png)

> 둔

### 5) Polyhedra

### 6) NormBall & NormCones