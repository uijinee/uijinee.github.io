---
title: "3. Convex Optimization"
date: 2024-04-06 22:00:00 +0900
categories: ["Math", "Convex Optimization"]
tags: ["math"]
use_math: true
---

# Convex Optimization

## 1. 정의

### 1) Standard Form
> 
> | | Optimization | Convex Optimization |
> | --- | --- | --- |
> | Miminize<br>_(목적함수)_ | $f_0(x), \quad x \in \mathbb{R}^n$ | $f_0(x)$ |
> | Subject to | $f_i(x) \leq 0$<br> $h_i(x) = 0$ | $f_i(x) \leq 0$<br>$a_i^T\mathbf{x}=b_i$ |
> | 설명 | **Constraint**에는 <br>ⅰ. Inequality Constraints<br> ⅱ. Equality Constraints<br>만 있어야 함 | ⅰ. $f_i(x), \quad i=0, 1,...,m$<br> $\quad \Rightarrow$ Convex Function이어야 함<br> ⅱ. Equality Constraints<br>$\quad \Rightarrow$ Affine Function이어야 함<br>$\quad\quad$_(Hyperplane)_ |
> 
> ---
> $x$가 Standard Form을 만족할 때$(x \in (\mathcal{D}(f_0) \cap  \mathcal{D}(f_i) \cap \mathcal{D}(h_i)))$,<br>
> **Feasible**하다고 정의한다.
> 
> 또 $f_0(x^\*)$가 $f(x)$의 Infimum일 때, $x^*$를 Optimal Value라고 한다.
> 
> &#8251; Minimum: 최솟값, Infimum: Lower Bound

### 2) Optimal Condition

> #### First Order Condition
> 
> 다음을 만족하는 점$x^*$를 Optimal이라고 정의한다.
>
> $$
> \nabla f_0(x^*)^T(x-x^*) \geq 0 \\
> $$
>
> 즉, 어느 방향으로 움직이더라도 $f$의 변화율이 양수인, 즉 $f$가 증가하는 점을 말한다.

---
## 2. 기본 형태

### 1) Unconstrained Problem

| Minimize | Subject To|
| --- | --- |
| $f_0(x)$ | $x \in \mathbb{R}^n$ |

> 가장 기본적인 형태로, Constraints에 대한 별다른 제약조건이 없을 경우 미분해서 0되는 점을 찾는다.
>
> $$
> \nabla f_0(x^*) = \textbf{0} \quad(\text{zero vector})
> $$
>
> _(위의 Solution이 가능한 이유는 $f_0$가 Convex함수이기 때문이다.)_

### 2) Equality Constrained Problem

| Minimize | Subject To|
| --- |:---:|
| $f_0(x)$ | $A\mathbf{x} = b$<br> _(p개의 equality constrained)_<br> _(Hyperplane의 교선)_ |

> Constraints에 대해 Equality Constraints(Affine Function)만 존재할 경우<br>
> **Dual Variable**($v$)을 도입함으로써 문제를 해결할 수 있다.
>
> $$
> \nabla f_0(x^*) + A^T\mathcal{v} = 0
> $$
>
> - $x^*$: Primary Variable
> - $v$: Dual Variable
>
> 즉, 다음의 두 식을 모두 만족하는 점을 찾아야 한다.
> - $\nabla f_0(\mathbf{x}^*) = -A^Tv$
> - $A\mathbf{x}^* = b$

### 3) Positive Orthant Problem

| Minimize | Subject To|
| --- |:---:|
| $f_0(x)$ | $\mathbf{x} \geq 0 $ |

&#8251; Positive Orthant: 1사분면

> 정의역이 항상 양수임이 보장되면 First Order Condition을 이용하여 문제를 해결할 수 있다.
>
> $$
> \nabla f_0(x^*)^Tx^* = 0
> $$
>
> 이는 $\nabla f_0(x^\*)^T(x-x^\*) \geq 0 \quad \forall x \geq 0$ 이므로,<br>
> - $x=2x^\* \rightarrow \nabla f_0(x^\*) \leq 0$
> - $x=0.5x^\* \rightarrow \nabla f_0(x^\*) \geq 0$
>
> 이 모두 성립해야 하기 때문이다.<br>
> 즉, 다음의 3 식을 모두 만족하는 점을 찾아야 한다.
>
> - $x^\* \geq 0$
> - $\nabla f_0(x^\*) \geq 0$
> - $\nabla f_0(x^\*)_i x_i^\* = 0$

---
## 3. Equivalent 형태

### 1) Equality Constraints 변형

> #### 제거
> 
> | | Minimize | Subject To|
> | --- | --- |:---:|
> | Base Form | $f_0(x)$ | $f_i(\mathbf{x}) \leq 0$<br>$A\mathbf{x} = b$ |
> | Equivalent Form | $f_0(F\mathbf{z}+\mathbf{x}_0)$ | $f_0(F\mathbf{z}+\mathbf{x}_0) \leq 0$ |
> 
> $A\mathbf{x} = b$의 해는 Homogeneous Solution($A\mathbf{x} = 0$)과 Particular Solution($A\mathbf{x} = b$)의 합으로 나타낼 수 있다.
> _($\mathbf{x} = \mathbf{x}_h + \mathbf{x}_p$)_
>
> 즉, $\mathbf{x} = F\mathbf{z} + \mathbf{x}_0$
> - $F$: $A$의 Null Space의 Basis(A의 벡터를 0으로 만들어 주는 공간)
> - $\mathbf{z}$: 임의의 벡터($\in \mathbb{R}^{dim(N(A))}$)
> - $\mathbf{x}_0$: Particular Solution의 해
> 
> ---
> #### 생성
>
> | | Minimize | Subject To|
> | --- | --- |:---:|
> | Base Form | $f_0(A\mathbf{x}+b_0)$ | $f_i(A_i\mathbf{x}+b_i) \leq 0$ |
> | Equivalent Form | $f_0(y_0)$ | $f_i(y_i) \leq 0$<br> $y_i = A_i\mathbf{x} + b_i$ |
>
> 마찬가지로 Constraints를 더 생성하는것이 도움이 될 때도 있다.

### 2) Slack 도입

### 3) 기타
> #### Epigraph Form
> #### Minimizing over some variables

---
## Optimal Solution구하기 (QP, QCQP)

## LP QCQP로 바꾸기

