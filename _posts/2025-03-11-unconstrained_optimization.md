---
title: "5-1. Unconstrained Optimization"
date: 2025-03-10 12:00:00 +0900
categories: ["Math", "Convex Optimization"]
tags: ["math"]
use_math: true
---


> J. Nocedal and S. J. Wright. Numerical Optimization. Springer, New York, 2nd edition, 2006.

# Unconstrained Optimization

# 문제 정의

$$
\min_x f(x), \\
x \in \mathbb{R}^n, n \leq 1, f: \mathbb{R}^n \rightarrow \mathbb{R}
$$

- 목표: 컴퓨터의 제한된 Resource를 활용해 주어진 식의 최적해를 찾는 알고리즘을 개발하는 것
- 문제점<br>
ⅰ) f의 전체적인 형태는 알 수 없다.<br>
ⅱ) f의 국소적인 정보는 알 수 있지만 이를 최소한으로 활용해야 한다.<br>
ⅲ) f에 noise가 있을 수 있기 때문에 최적점에 대한 정의가 필요하다.<br>

※ 예를 들어 $f(x) =(Y - \phi(\mathbf{t};\mathbf{x}))^2$ ($Y$는 관측값, $\phi(\mathbf{t})$는 예측값, x는 모델의 파라미터)라고 하자. 이때, 우리가 원하는 것은 $f(x)=0$이지만 noise나 파라미터의 부족 등 다양한 이유로 실제로는 $f(x)=0$을 만족하지 못한다.

즉, 우리는 들어가기에 앞서 최적점에 대한 정의를 내릴 필요가 있다.

# 최적점

## 1. 정의

### 1) Global minima

$$
f(x^*) \leq f(x)
$$

즉, 이 함수에서 가장 작은 점을 말한다.

### 2) local minima

$$
f(x^*) \leq f(x), \qquad x \in \mathcal{N}
$$

즉, 주변의 점(양 옆)에 대해 제일 작은 점을 말한다.

---
## 2. 판별법

### 1) Taylor's theorem
 
> $f$가 연석이고 미분 가능하면 다음이 성립한다.
> 
> $$
> f(x + p) = f(x) + \nabla f(x + tp)^Tp, \qquad t \in (0, 1)
> $$
> 
> ※ 위 정리는 평균값 정리에 의해 유도될 수 있다. <br>
> $g(t) = f(x + tp)$라고 할 때, 평균값 정리에 의해 $g(1) - g(0) = g'(t)(1-0), \quad t \in (0, 1)$을 만족하는 값이 반드시 존재한다.
>
> 이때, $f$가 두번 미분 가능하다고 가정하면 다음 수식이 만족된다.
> 
> $$
> \nabla f(x+p) = \nabla f(x) + \int^1_0 \nabla^2f(x + tp)p dt, \qquad t \in (0, 1)
> $$
>
> ※ 마찬가지로 $\psi(t) = \nabla f(x + tp)$로 정의하자. 이때, 기본 정리에 의해 $\psi(1) = \psi(0) + \int^1_0 \psi'(t) dt$이므로 이를 정리하면 위의 수식을 얻을 수 있다.
>
> $$
> f(x + p) = f(x) + \nabla f(x)^Tp + \frac{1}{2} p^T \nabla^2 f(x+tp)p, \qquad t \in (0, 1)
> $$
>
> ※ 이 부분은 잘 모르겠다...
>

### 2) First-order Necessary conditions

> $$
> x^* \text{ is local minima}, f \text{ is 미분가능\연속} \rightarrow \nabla f(x^*) = 0
> $$
>

### 3) Second-order Necessary conditions

> $$
> x^* \text{ is local minima}, \nabla^2 f \text{ is 존재\연속} \rightarrow \nabla^2 f(x^*) \succeq 0 \; , \nabla f(x^*) = 0
> $$

### 4) Second-order Sufficient conditions

> $$
> \nabla^2 f \text{ is 존재}, \nabla f(x^*) = 0, \nabla^2 f(x^*) \succeq \rightarrow x^* \text{ is strict local minima}
> $$

### 5) Convex function

> $$
> f \text{ is convex}, \nabla f(x^*) = 0 \rightarrow x^* \text{ is global minima} \\
> $$


---
# 알고리즘

위에서 최적점을 정의하고 어떤 점이 최적점인지 알아보았다. 그러면 이제 이 최적점을 어떻게하면 찾을 수 있을지 알고리즘을 알아보자.

들어가기에 앞서 이 점을 찾는 알고리즘은 크게 두가지로 분류된다. 이때 이 두가지 알고리즘은 모두 다음과 같은 특징을 가지고 있다.

- 시작점에 대한 정의가 필요하다. 이것은 사용자나 알고리즘을 통해 결정한다.
- 시작점에서 시작해 Iterative한 과정으로 최적점을 찾아 나간다.
- $x_k$에서 $x_{k+1}$로 이동하는 알고리즘이다. 이때, $f(x_k) > f(x_{k+1})$가 반드시 성립할 필요는 없지만, 정해진 구간안에 감소해야한다.

위의 공통점을 가지면서 이동 방식(이동 방향, 이동 거리)에 따라 다음과 같이 알고리즘을 나눌 수 있다.<br>
또한 이 알고리즘이 찾아야 하는 것은 <u>이동 방향과 이동 거리</u>임을 명심하자.

| Line Search | Trust region |
| --- | --- |
| ⅰ. 방향($p_k$)을 찾아 고정한다.<br> ⅱ Step size($\alpha_k$)를 찾는다.<br> ⅲ. $\alpha_k$가 조건을 만족하는지 판별한다.<br> ⅳ. 불만족시 (ⅱ)과정을 다시 수행한다.<br> ⅴ. 위 과정을 반복한다. | ⅰ. Maximum stepsize $\Lambda_k$를 설정한다.<br>$\quad =$ trust region <br> ⅱ. 이 조건 하에서 방향과 Step size를 찾는다.<br> ⅲ. 조건을 불만족할 시 $\Lambda_k$를 줄인다.<br> ⅳ. 위 과정을 반복한다. |

즉, Line search는 방향을 먼저 선택하여 고정한 상태에서 이동 거리를 찾아 움직이는 방식이다. 하지만 trust region 방식은 maximum 이동 거리를 찾고 이 안에서 방향과 이동 거리를 모두 찾는다는 차이점이 있다. 이 차이점을 명심하며 각 알고리즘에 대해 더 자세히 알아보자.

## 1. Line search

$$
\min_{\alpha > 0} f(x_k + \alpha_k p_k)
$$

- 정의<br>
Line search알고리즘은 위와 같이 특정 방향($p_k$)으로 특정 거리($\alpha_k$)만큼 움직였을 때, 어떻게 하면 가장 작아질 수 있는지를 찾는 알고리즘이다. 

- 종류<br>
위의 식에서 방향을 정하는 방식에 따라 다음과 같이 나눌 수 있다. (Step size를 설정하는 방식은 <mark>후술 예정<\mark>)
    - Steepest descent
    - Newton's method
    - Quasi-newton's method

### 1) Steepest decent

> 현재 점($k$)에서 가장 가파르게 감소하는 방향인 $p_k = - \nabla f_k$로 이동 방향을 설정하는 방식이다.
>
> - 장점: 미분값만 계산하면 되기 때문에 계산 속도가 빠르다.
> - 단점: 수렴 속도가 느리다.
>
> **※ 이 방향이 왜 가파른 방향이고 감소하는 방향일까?**<br>
> 이를 증명하기 위해서는 Taylor's theorem(2-1)은 다음과 같이 다시 표현할 수 있다.
> 
> $$
> f(x + p) = f(x) + \alpha \nabla f(x)^Tp + \frac{1}{2} \alpha^2 p^T \nabla^2 f(x+tp)p
> $$
> 
> 이때, 현재 점에서 가장 가파른 방향을 찾아야 하므로 $\alpha_k$가 매우 작다고 하면 다음이 성립한다.
> 
> $$
> f(x + p) \approx f(x) + \alpha \nabla f(x)^Tp
> $$
>
> 즉, Line search의 목적함수는 다음과 같이 변경할 수 있다.
> 
> $$
> \min_p p^T \nabla f_k \qquad , \text{subject to } \Vert p \Vert = 1
> $$
>
> 이때, $p^T\nabla f_k = \Vert p \Vert \Vert \nabla f_k \Vert cos \theta = \Vert \nabla f_k \Vert cos \theta$ 이므로 $p = - \frac{f_k}{\Vert \nabla f_k \Vert}$ 라고 할 수 있다.
>
> ※ 참고로 $p_k$와 $-\nabla f_k$의 각도가 $\frac{\pi}{2}$미만일 경우에도 f가 감소하는 방향이라는 것은 보장된다.
>

### 2) Newton's method

> Taylor근사를 사용해 이동 방향을 $p_k = - (\nabla^2 f_k)^{-1}\nabla f_k$로 설정하는 방식이다.
>
> - 장점: 수렴 속도가 제곱으로 빨라진다.
> - 단점: Hessian을 계산해야하기 때문에 많은 비용이 발생한다.
>
> **※ Taylor 근사의 minima가 감소하는 방향인 이유?**<br>
> Line search의 목적 함수는 Taylor 근사를 통해 다음과 같이 표현할 수 있다.
> 
> $$
> f(x_k+p_k) \approx f_k + p^T \nabla f_k + \frac{1}{2} p^T \nabla^2 f_k p \overset{\text{def}}{=} m_k(p)
> $$
>
> 여기서 $\nabla^2 f_k \succ 0 $를 가정하면 $f(x_k + p_k) \approx m_k(p)$의 global minimum은 단순히 $\nabla m_k(p) = 0$인 점을 찾음으로써 구할 수 있다. 즉, $p_k = - (\nabla^2 f_k)^{-1}\nabla f_k$ 이다.<br>
> 
> Steepest descent와 마찬가지로 Line search의 목적함수는 $p^T \nabla f_k$를 작게 만드는 것으로 근사할 수 있고, 이 값이 음수일 때 f가 감소한다는 것을 알 수 있다. 이때, $p_k = - (\nabla^2 f_k)^{-1}\nabla f_k$일 경우, $\nabla f_k^T p_k = -p_k^T \nabla^2 f_k p_k \leq -\sigma_k \Vert p_k \Vert^2$가 성립한다.
> 
> 이때, $\nabla^2 f_k \succ 0$을 가정하였으므로 $\sigma_k > 0$이고 $\nabla f_k^T p_k \leq 0$이 항상 성림한다는 것을 알 수 있다.
> 
> ※ $\nabla^2 f_k \succ 0$이 아닐 경우 Newton's method는 성립되지 않는다. 이 때는 $\nabla^2 f_k^{-1}$이 존재하지 않을 수 있고, 만약 존재한다 해도 $\nabla f_k^T p_k^N < 0$ 이 성립하지 않을 수 있기 때문이다. 이러한 상황을 해결하기 위해 실제 알고리즘들은 몇가지 수정 작업을 거친다. <mark>(후술 예정)</mark>
> 
> ※ Steepest decent와 다르게 Newton's method에서는 step size를 보통 1로 설정한다. 이때, 결과가 만족스럽지 않을 경우에 step size를 줄이는 방식으로 구현한다.

위에서 보았다 싶이 Newton's method는 매력적인 방법이지만 계산비용이 크다는 단점이 있었다. 이에 이를 해결하기 위한 여러가지 방법이 제안 되었고 이 중 Quasi-newton's method 방법을 알아보자.

### 3) Quasi-Newton's method

Quasi-newton 방식은 Hessian $\nabla^2 f_k$을 직접 구하지 않고 $B_k$라는 근사 행렬을 사용한다. 이때 $B_k$는 각 step마다 업데이트 된다.

이 방식에 대한 핵심 idea는 1차 도함수에 대한 변화를 통해 2차 도함수(Hessian)에 대한 정보를 추정할 수 있을 것이라는 점에서 시작된다.

먼저 Taylor's theorem에서 다음과 같은 수식을 유도할 수 있다.

$$
\nabla f(x+p) = \nabla f(x) + \int^1_0 \nabla^2f(x + tp)p dt \\ 
= \nabla f(x) + \int^1_0 [\nabla^2f(x + tp)p - \nabla^2 f(x) ]dt + \int ^1_0 \nabla^2 f(x) p dt \\
= \nabla f(x) + \nabla^2 f(x) p + \int^1_0 [\nabla^2f(x + tp)p - \nabla^2 f(x) ]dt
$$

즉, $\nabla f(x_{k+1}) = \nabla f(x_k) + \nabla^2 f(x_k) (x_{k+1}-x_k) + o(\Vert x_{k+1} - x_k \Vert)$라는 것을 알 수 있고, 결국 1차도함수의 변화량은 $\nabla^2 f(x_k) (x_{k+1}-x_k)$과 근사된다는 것을 알 수 있다.

$$
\nabla^2 f_k(x_{k+1} - x_k) \approx \nabla f_{k+1} - \nabla f_k 
$$

즉, 현재 $B_k$가 있다고 하면, 이 값은 이것을 $B_{k+1} s_k = y_k$를 활용해 업데이트할 수 있게 된다. 이때, 보통의 Hessian이 symmetry하다는 점에서 착안해 보통 $B_k$는 symmetry한 행렬로 설정한다.

또 위 형태는 [secant equation](https://en.wikipedia.org/wiki/Secant_method)으로 표현할 수 있다는 것을 알 수 있다.<br>
우리는 $B_{k+1} = B_{k} + U$ 방식으로 업데이트 하면서, 이와 동시에 업데이트 이후에도 "symmetric"이라는 점과 secant equation($(B_k + U)s_k = y_k$)을 만족하도록 하고싶다.<br>
이를 위해서 사용할 수 있는 공식은 다음이 있다.

- 방법1) symmetric-rank-one(SR1) formula<br>
: $B_{k+1} = B_k + \frac{(y_k - B_ks_k)(y_k - B_ks_k)^T}{(y_k - B_ks_k)^Ts_k}$
- 방법2) BFGS formula(rank-2 matrix)<br>
: $B_{k+1} = B_k - \frac{B_ks_ks_k^TB_k}{s_k^TB_ks_k} + \frac{y_ky_k^T}{y_k^Ts_k}$

([증명](https://convex-optimization-for-all.github.io/contents/chapter18/2021/03/23/18_02_Symmetric_Rank_One_Update_(SR1)/): 이 부분은 아직 잘 이해가 안된다.)

※ 실제 구현할 때, 몇몇 방식은 B를 업데이트하는게 아니라 $B^{-1}$을 업데이트하는 방식을 사용해 더 간단하게 식을 구성하는 경우도 있다.

### 기타

- Nonlinear conjugate gradient method<br>
: $p_k = -\nabla f(x_k) + \beta_k p_{k-1}$


---
---
## 2. Trust regioin

(pass)
