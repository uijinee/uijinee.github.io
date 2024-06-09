---
title: "5. Algorithm"
date: 2024-05-16 22:00:00 +0900
categories: ["Math", "Convex Optimization"]
tags: ["math"]
use_math: true
---


## 1. Unconstrained Minimization

$$
\text{Minimize: } f(\mathbf{x}) 
$$

$\mathbf{x} \in \mathbb{R}^n,$<br>
$f(\mathbf{x}) \text{는 Convex},$<br> 
$f(\mathbf{x}) \text{는 두번 미분 가능}$

> #### Background
> 
> Minimization을 Algorithm으로 구현할 때의 핵심은 $f(x^{(n)}) = p^\*$일 때, Iterative Method를 사용해 KKT Condition을 만족할 때 까지 $x^{(0)}, x^{(1)}, ... , x^{(n)}$점으로 진행하는 것이다.
>
> 이때, Unconstrained문제이므로 이 최적점 $\mathbf{x}^{(n)}$는 결국 $\nabla f(\mathbf{x}^\*) \approx 0$을 만족할 것이다.
>
> 이 점을 찾기 위해서 Descent Method를 사용할 수 있는데, 이 방법의 핵심은 다음과 같다.<br>
> 먼저 Convex함수이기 때문에 항상 내려가는 방향으로 움직이기만 하면 된다.
> 
> $$
> \mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + t^{(k)} \vartriangle \mathbf{x}^{(k)} \\
> f(\mathbf{x}^{(k)}) > f(\mathbf{x}^{(k+1)})
> $$
> 
> $\vartriangle \mathbf{x}$는 방향<br>
> $t > 0$는 크기
>
> 즉, 위 식을 만족하는 $f(\mathbf{x}^{(k+1)})$을 찾아야 하는데, 이를 찾기 위해 다음의 사실을 이용할 수 있다.<br>
> $f(\mathbf{x}^{(k+1)}) > f(\mathbf{x}^{(k)}) + \nabla f(\mathbf{x}^k)(\mathbf{x}^{k+1} - \mathbf{x}^k) \rightarrow$ _(Convex의 정의)_<br>
>
> 즉, 해당 지점에서의 미분값과 크기를 $\nabla f(\mathbf{x}^{(k)})^T \vartriangle \mathbf{x}^{(k)} < 0$를 만족하도록 정의하면 된다.
>
> ---
> #### Line Search
>
> | Exact Line Search | Backtracking Line Search |
> | --- | --- |
> | $t = \text{arg} \min \limits_{t > 0} f(\mathbf{x} + t \vartriangle \mathbf{x})$ | |
> | t에 대해 Restriction to a line을 수행해 <br> t에 대해 최소화 하는 것<br> _(Restriction to a line이기 때문에 Convex임)_|  |
> | 단점: Convex문제를 풀어야하기 때문에 복잡도 $\Uparrow$ |  |
> 
> t = 


### 1) Gradient Descent

> 이제 이동할 크기를 정하는 방법을 먼저 살펴보자

이동할 방향: gradient descent

Strongly Convex일 경우 Convergincse가 보장됨
※ Strongly Convex: $\nabla f(\mathbf{x}) - mI\succeq 0$ (휘어짐의 정도(곡률)이 $mI$보다 큼)


### 2) Steepest Descent

### 3) Newton's Method

### 1) 