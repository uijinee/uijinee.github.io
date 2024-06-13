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

### 1) Equality Constrained Optimization

| Problem | KKT Condition |
| --- | --- |
| $$ \text{Minimize: } \quad f(\mathbf{x}) \\ \text{Subject to}: \quad A\mathbf{x} = \mathbf{b} $$<br><br>  $A \in \mathbb{R}^{p \times n},$<br> $Rank(A) = p$<br> $\mathbf{x} \in \mathbb{R}^n,$<br> $f(\mathbf{x}) \text{는 Convex},$<br> $f(\mathbf{x}) \text{는 두번 미분 가능}$ | $$ \text{Primal Feasible: } \; \qquad A\mathbf{x}^* = \mathbf{b}$$<br> $$\text{Gradient of Lagrangian: } \quad \nabla f(\mathbf{x}^*) + A^T\nu^* = 0 $$<br><br><br> &#8251; Dual Residual: $r_d(\mathbf{X}, \nu) = \nabla f(\mathbf{x}) + A^T\nu$<br> &#8251; Primal Residual: $r_p(\mathbf{x}, \nu) = A\mathbf{x} - \mathbf{b}$  |

&#8251; Equality Constrained Quadratic Minimization일 경우 주의할 점이 필요하므로 이를 살펴보자.

| Problem | KKT Matrix |
| --- | --- |
| $$\text{Minimize: } \quad \frac{1}{2} \mathbf{x}^{*T}P\mathbf{x}^* + q^T\mathbf{x}^* + r$$<br> $$\text{Subject to}: \; A\mathbf{x} = \mathbf{b}$$<br><br>-----------------------------------------------------<br>**KKT Condition**<br> $$P\mathbf{x}^* + q + A^T \nu = 0$$ <br>$$A\mathbf{x}^* = \mathbf{b}$$<br> $$\qquad \Downarrow$$<br> $$\begin{bmatrix} P & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} \mathbf{x}^* \\ \mathbf{\nu}^* \end{bmatrix} = \begin{bmatrix} -q \\ \mathbf{b} \end{bmatrix}$$| $$K = \begin{bmatrix} P & A^T \\ A & 0 \end{bmatrix}$$<br> 이때 문제를 풀기위해 KKT Matrix가<br> Non-Singular Matrix(역함수 존재)임을 가정하자.<br>$\Rightarrow \mathcal{N}(P) \cap \mathcal{N}(A) = \begin{Bmatrix} 0\end{Bmatrix}$<br>$\Rightarrow \mathbf{x}^TP\mathbf{x} + \Vert A\mathbf{x} \Vert^2 > 0 \quad \because$(P:정방행렬, A:비정방행렬)<br>$\Rightarrow P + A^TA \succ 0$<br>즉, 위 식이 성립하면 Unique한 Solution을 구할 수 있다.<br> _(또 만약 $P \succ 0$일 경우에도 마찬가지이다.)_ |

> #### Eliminating Equality Constraint
>
> KKT Condition으로도 문제를 풀 수는 있다.<br>
> 
> 하지만 앞서 배운 Unconstrained Optimization에서 사용하는 알고리즘들(Gradient Descent, Newton's Method)를 사용하기 위해서는 Constraint를 없애줄 필요가 있다.
>
> | | $\mathbf{x} = \mathbf{x}_h + \mathbf{x}_p$ |
> | --- | --- |
> | ![alt text](/assets/img/post/convex_optimization/nonhomogeneous_solution.png) | Nonhomogeneous System에서 배웠듯이 $A\mathbf{x} = \mathbf{b}$의 해는<br> Homogeneous Solution($A\mathbf{x} = 0$)과<br> Particular Solution($A\mathbf{x} = \mathbf{b}$)으로 나눌 수 있다. |
> 
> 이를 이용하여 Constraint가 존재하는 $\mathbf{x}$는 Constraint가 존재하지 않는 $\mathbf{z}$에 대한 식으로 바꿀 수 있다.
>
> | $$ \qquad \begin{Bmatrix} \mathbf{x} \vert A \mathbf{x} = b \end{Bmatrix} \rightarrow \begin{Bmatrix} F\mathbf{z} + \hat{\mathbf{x}} \vert \mathbf{z} \in \mathbb{R}^{n-p} \end{Bmatrix} \\ \, \\F \in \mathbb{R}^{n \times (n-p)} \\ AF = 0, \quad (\mathcal{N}(A) = F)\\ A\mathbf{\hat{x}} = \mathbf{b} \qquad \qquad \qquad \quad$$ | $$\Rightarrow \qquad \underset{z}{\text{Minimize}} \quad f(F\mathbf{z} + \hat{x}) \qquad \qquad $$ |
>
> 즉 이 Unconstrained Minimization Problem의 해 $\mathbf{z}^\*$를 구하고<br>
> $\mathbf{x}^\* = F\mathbf{z}^\* + \hat{\mathbf{x}}, \quad \nabla f(\mathbf{x}^*) + A^T \nu^\* = 0$를 통해 원래의 해($\mathbf{x}^\*, \nu^\*$)를 알 수 있다.
>
> ---
> #### Projected Gradient Method
>
> ![alt text](/assets/img/post/convex_optimization/projected_gradient_method.png)
>
> 다음 3개의 문제를 살펴보자.
>
> | Problem1 | Problem2 | Problem3 |
> | --- | --- | --- |
> | $$ \underset{\mathbf{x}}{\text{Minimize}} \quad f(\mathbf{x}) \\ \text{Subject to} \quad A\mathbf{x} = \mathbf{b} $$ | $$\underset{z}{\text{Minimize}} \quad f(F\mathbf{z} + \hat{x})$$ | $$\underset{\mathbf{y}}{\text{Minimize}} \quad \Vert F\mathbf{y} - (-g) \Vert_2$$ |
> | --- | $$\mathbf{z}^* = F^T \cdot \nabla_\mathbf{x} f(F\mathbf{z} + \hat{\mathbf{x}}) \\ \vartriangle\mathbf{z} = -\mathbf{z}^*$$ | $$\mathbf{y}^* = -F^Tg$$<br>_(if $F$ is Orthonormal basis)_ |
>
> 우선 1번과 2번이 같은 문제임은 앞에서 증명하였다.
>
> 이때, 3번에서 $g= \nabla_\mathbf{z} f(F\mathbf{z} + \hat{\mathbf{x}})$라고 설정하면 2번문제의 Gradient Descent의 Step이 3번 문제의 Optimal인 점과 동일하다는 것을 알 수 있다.
>
> 즉, $g = \nabla_\mathbf{x} f$를 $F$가 Span하는 공간 위로 Projection시킨 벡터가 해임을 알 수 있다.
>
> 이를 통해 알 수 있는 것은 Equality Constrained일 경우,<br>
> Unconstrained일때의 Gradient Descent의 Step은 Equality Constrained일 때<br>
> $A$의 Null Space(즉, Homogeneous해) 평면 위로<br>
> Projection되어 진행한다는 것이다.
> 
> ---
> #### Newton Step
>
> Newton Step은 현재 위치에서 2차근사를 활용해 최솟값을 찾는 방식이다.<br>
> 즉, 다음의 문제를 풀면된다.
> 
> | Problem | KKT Condition |
> | --- | --- |
> | $$\underset{\mathbf{\nu}}{\text{Minimize:}}\; \hat{f}(\mathbf{x} + \mathbf{\nu}) = f(\mathbf{x}) + \nabla_\mathbf{x} f(\mathbf{x})^T \nu + \frac{1}{2} \nu^T \nabla^2_\mathbf{x} f(\mathbf{x}) \nu \\ \text{Subject to: }\quad A(\mathbf{x} + \nu) = \mathbf{b}$$| **ⅰ. Primal Feasible**<br> $$A(\mathbf{x} + \nu) = \mathbf{b}$$ <br>**ⅳ. Gradient of Lagrangian**<br>$$\nabla_\nu (\hat{f}(\mathbf{x} + \mathbf{\nu}) + A(\mathbf{x} + \nu) - \mathbf{b}) \\ = \nabla_\mathbf{x} f(\mathbf{x}) + \nabla^2_\mathbf{x}f(\mathbf{x}) \nu + A^Tw = 0$$ |
> 
> 즉 다음의 Linear Equation의 Solution이다.
>
> $$
> \begin{bmatrix} \nabla^2_\mathbf{x} f(\mathbf{x}) & A^T \\ A & 0\end{bmatrix} \begin{bmatrix} \nu \\ W \end{bmatrix} = \begin{bmatrix} -\nabla_\mathbf{x} f(\mathbf{x}) \\ 0 \end{bmatrix}   
> $$
>
> &#8251; Newton Decrement: $$\lambda(\mathbf{x}) = (\vartriangle \mathbf{x}^T_{nt} \nabla^2 f(\mathbf{x}) \vartriangle \mathbf{x}_{nt})^\frac{1}{2} = (-\nabla f(\mathbf{x})^T \vartriangle \mathbf{x}_{nt})^\frac{1}{2}$$

### 2) Inequality Constrained Optimization

Inequality로 바꾸기