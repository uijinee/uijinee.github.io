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
> 이제 이동할 크기(Learning Rate)를 정하는 방법을 먼저 살펴보자
>
> | Exact Line Search | Backtracking Line Search |
> | --- | --- |
> | $t = \text{arg} \min \limits_{t > 0} f(\mathbf{x} + t \vartriangle \mathbf{x})$ | $f(\mathbf{x} + t \vartriangle \mathbf{x}) < f(\mathbf{x}) + \alpha t \nabla f(\mathbf{x})^T \vartriangle \mathbf{x}$ |
> | t만큼 움직였을 때, 함수값이 가장 작아질 때 | 현재위치에서 접선의 기울기보다<br> 기울기의 절댓값을 줄인 직선보다<br> 아래에 있을 때 |
> | t에 대해 Restriction to a line이기 때문에 <br> Convex함수 이므로 Minimization문제를 푼다 | ![alt text](/assets/img/post/convex_optimization/backtracking_linesearch.png)<br> ⅰ. $t=1, \alpha \in (0, \frac{1}{2}), \beta \in (0, 1)$로 Setting<br>ⅱ. $t = \beta t$를 위 식을 만족할 때 까지 반복 |
> | **단점**<br>: Convex문제를 풀어야하기 때문에 복잡도 $\Uparrow$ | 생각보다 성능이 좋다. |

위에서는 learning rate들을 정할 수 있는 몇가지 방법을 알아보았으니 이제 방향 등 자세한 방식들을 알아보자.

### 1) Gradient Descent

| Gradient Descent | Steepest Descent |
| --- | --- |
| | _(Gradient Descent의 일반화 ver)_ |

$\mathbf{x}$에서 움직일 방향을 Gradient로 설정하는 것

> #### Algorithm
> 
> ```python
> while stop:
>     x = now
>     direction = -gradent(f, x)
>     step_size = line_search(f, x, alpha, beta)
>     now = x + step_size * direction
> ```
> 
> ---
> #### Convergence rate
>
> $f(\mathbf{x})$가 Strongly Convex일 경우 다음 식을 항상 만족한다.
>
> $$
> f(\mathbf{x}^{(k)}) - p^* \leq c^k (f(\mathbf{x}^{(0)}) - p^*)
> $$
>
> 즉, Convergence가 보장된다.<br>
> _(c가 작을수록 Convergence 속도는 느려짐)_
> 
> ※ Strongly Convex: $\nabla f(\mathbf{x}) - mI\succeq 0$ <br>
> $\quad$(휘어짐의 정도(곡률)이 $mI$보다 큼)
>
> ---
> #### 단점
>
> Gradient Descent는 현재 위치에서의 기울기, 즉 Local한 정보만을 활용하기 때문에 수렴 속도가 매우 느리다.
>
> $$ 
> f(\mathbf{x}) = \mathbf{x}_1^{2} + \gamma \mathbf{x}_2^2, \gamma > 0 \\
> \mathbf{x}^{(0)} = (\gamma, 1) \quad \text{& Exact Line Search}
> $$
>
> | ![alt text](/assets/img/post/convex_optimization/bad_gradient_descent.png) | $$\therefore \mathbf{x}_1^{(k)} = \gamma(\frac{\gamma - 1}{\gamma + 1})^k, \quad \mathbf{x}_2^{(k)} = (-\frac{\gamma - 1}{\gamma + 1})^k$$ |
> 
> $\qquad \Rightarrow$ Oscillate 발생


### 2) Newton's Method


---
## 2. Constrained Minimization

핵심: Unconstrained로 바꾸기