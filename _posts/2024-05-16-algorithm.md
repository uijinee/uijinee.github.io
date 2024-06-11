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

$\mathbf{x}$에서 움직일 방향을 Gradient로 설정하는 방식이다.

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + t^{(k)} \vartriangle \mathbf{x}^{(k)}$$

| | Gradient Descent | Steepest Descent |
|:---:|:---:|---|
| $\vartriangle x$ | $\vartriangle x = -\nabla f(\mathbf{x})$ | **ⅰ. Normalized**<br> $\quad \vartriangle x_{nsd} = \text{arg}\min \limits_{\Vert v \Vert_P \leq 1} (\nabla f(\mathbf{x})^T v)$<br> $\quad$_(Gradient Descent의 일반화 ver)_<br> $\quad$_(v에 대한 Norm)_<br><br> **ⅱ. Unnormalized**<br> $$\quad \vartriangle x_{sd} = \Vert \nabla f(\mathbf{x}) \Vert_* \vartriangle \mathbf{x}_{nsd}$$<br>$\,$ |
| 목표 | 현재 기울기가 가장 가파른 곳으로 이동<br> Unnormalized Steepest Descent의<br> 2-norm ver | $f(\mathbf{x} + v) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^Tv$<br> $v$만큼 움직였을 때<br> 함숫값이 최소가 되도록<br>$\,$ |
| 해법 | 현재 위치에서 Jacobian을 구한다. | $$\text{Minimize} \qquad \nabla f(\mathbf{x})^T v \\ \text{Subject to} \qquad \Vert v \Vert_P \leq 1$$<br> 문제 이므로 Convex Optimization 문제 |

| | L1-Norm | Euclidean Norm | Quadratic Norm |
| --- | --- | --- | --- |
| $\vartriangle \mathbf{x}_{nsd}$ | | $$-\frac{\nabla f(\mathbf{x})}{\Vert \nabla f(\mathbf{x}) \Vert_2}$$ | |
| $\vartriangle \mathbf{x}_{sd}$ | | $$\Vert \nabla f(\mathbf{x}) \Vert_2 \times -\frac{\nabla f(\mathbf{x})}{\Vert \nabla f(\mathbf{x}) \Vert_2}$$<br> $$\therefore - \nabla f(\mathbf{x})$$ | $$\therefore -P^{-1}\nabla f(\mathbf{x})$$ |

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
> #### Algorithm
> 
> ```python
> while stop:
>     x = now
>     direction = -Gradent(f, x)
>     step_size = Line_search(f, x, alpha, beta)
>     now = x + step_size * direction
> ```
> 
> ---
> #### 단점
>
> | 수렴 속도 | P의 설정 |
> | --- | --- |
> |  Gradient Descent는 현재 위치에서의 기울기,<br> 즉 Local한 정보만을 활용하기 때문에<br> 수렴 속도가 매우 느리다. | P의 설정에 따라 수렴 속도가 매우 달라짐 |
>
> | 예시 | 그림 |
> |:---:| --- |
> | $$f(\mathbf{x}) = \mathbf{x}_1^{2} + \gamma \mathbf{x}_2^2, \gamma > 0$$<br> $$\mathbf{x}^{(0)} = (\gamma, 1) \quad \text{& Exact Line Search}$$ <br><br> $$\therefore \mathbf{x}_1^{(k)} = \gamma(\frac{\gamma - 1}{\gamma + 1})^k, \quad \mathbf{x}_2^{(k)} = (-\frac{\gamma - 1}{\gamma + 1})^k$$ | ![alt text](/assets/img/post/convex_optimization/bad_gradient_descent.png) |
> 
> $\qquad \Rightarrow$ Oscillate 발생

### 2) Newton's Method

| Interpretation | 그림 |
| --- | --- |
| **ⅰ Local Hessian Norm**<br> $\quad$ Gradient Descent는 Local한 정보($\nabla f(\mathbf{x})$)만<br> $\quad$ 이용했기 때문에 문제가 발생하였다.<br><br> $\quad$ 즉, Global한 정보인 Hessian을 고려해야 하고<br> $\quad$ 이를 위해 방향마다 거리를 다르게 설정하는<br> $\quad$ Quadratic Norm을 활용할 수 있다. <br><br> $$\quad \therefore P = \nabla^2 f(\mathbf{x})$$<br> $$\quad \rightarrow \vartriangle \mathbf{x}_{sd} = - (\nabla^2 f(\mathbf{x}))^{-1}(\nabla f(\mathbf{x})) $$ <br><br> **ⅱ. Taylor Series**<br> $\quad f(\mathbf{x})$를 $\mathbf{v}$만큼 움직였을 때의 함수를<br> $\quad$ Taylor Series의 2차항까지로 표현하면<br> $\quad$ 다음과 같다.<br>$$\quad f(\mathbf{x + v}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})\mathbf{v} + \frac{1}{2} \mathbf{v}^T \nabla^2 f(\mathbf{x})\mathbf{v}$$<br> $\quad$ 즉, 이 2차함수를 최소화하는 $\mathbf{v}$를 찾으면,<br> $\quad \mathbf{v} = \vartriangle \mathbf{x} = -\frac{\nabla f(\mathbf{x})}{\nabla^2 f(\mathbf{x})}$이다. | ![alt text](/assets/img/post/convex_optimization/newtons_method.png) |

Newton's Method는 위의 두가지 관점으로 해석할 수 있다.


> #### Newton Decrement (Stopping Criterion)
>
> Newton Decrement는 다음과 같이 정의된다.
> 
> $$
> \lambda(\mathbf{x}) = (\nabla f(\mathbf{x})^T \nabla^2f(\mathbf{x}^{-1}) \nabla f(\mathbf{x}))^{\frac{1}{2}} \\
> \quad = (\vartriangle \mathbf{x}_{nt}^T \nabla^2 f(\mathbf{x}) \vartriangle \mathbf{x}_{nt})^\frac{1}{2}
> $$
>
> 이때, Newton Decrement는 다음과 같은 특징을 갖는다.<br>
> $f(\mathbf{x}) - \inf \limits_y \hat{f}(y) = \frac{1}{2} \lambda(\mathbf{x})^2$
>
> 즉 이 Newton Decrement가 0 에 가깝다면 최적점에 가까워졌다고 할 수 있고<br>
> 따라서 이를 Stopping Criterion으로 사용할 수 있다.
> 
> ---
> #### Algorithm
> 
> ```python
> not_stop = True
> while not_stop:
>   x = now
>   x_nt = -Jacobian(f, x) / Hessian(f, x)
>   decrement = x_nt.T @ Hessian(f, x) @ x_nt
>   if (decrement / 2) < 0.01:
>       not_stop = False
>   step_size = Line_search(f, x, alpha, beta) 
>   now = x + step_size * x_nt
> ```
> 
> ---
> #### 특징
>
> ![alt text](/assets/img/post/convex_optimization/newtons_method_property.png)
> 
> - $\mathbf{x}$에 대한 함수를 $\mathbf{y} = T^{-1}\mathbf{x}$에 대한 함수로 Transform하더라도 Newton's Method를 사용해 이동할 경우 Mapping관계가 그대로 유지된다.

---
## 2. Constrained Minimization

핵심: Unconstrained로 바꾸기