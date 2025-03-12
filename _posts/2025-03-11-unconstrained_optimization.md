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

## 1. 최적점의 정의

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
## 2. 최적점 판별법

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
## @ 알고리즘

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

## 3. Line search 알고리즘

$$
\min_{\alpha > 0} f(x_k + \alpha_k p_k)
$$

- 정의<br>
Line search알고리즘은 위와 같이 특정 방향($p_k$)으로 특정 거리($\alpha_k$)만큼 움직였을 때, 어떻게 하면 가장 작아질 수 있는지를 찾는 알고리즘이다. 

- 방향을 정하는 방식들<br>
    - Steepest descent
    - Newton's method
    - Quasi-newton's method

- Step size를 정하는 방식들<br>
    - Backtracking line search

### --방향 설정 방식--

간단하게 요약하면 Line search의 경우 매 iteration마다 방향($p_k$)과 Step size($\alpha_k$)를 구해야 한다. 그리고 이를 조합해($x_{k+1} = x_k + \alpha_k p_k$) 다음 좌표를 계산한다. 여기서 유의할 점은 $p_k$는 $p_k^T \nabla f_k < 0$을 만족하는 값이어야 $f$를 감소시킬 수 있다는 점이다. 대부분의 알고리즘에서 이 $p_k$는 다음과 같은 형태를 갖는다.
 
$$
p_k = - B_k^{-1} \nabla f_k \\
B_k \text{ is symmetric and nonsingular matrix}
$$

- Steepest descent<br>
: $B_k = I$

- Newton's method<br>
: $B_k = \nabla^2 f(x_k)$

- Quasi-Newton's method<br>
: $B_k = B_{k-1} + U$

### 1) Steepest descent

> 현재 점($k$)에서 가장 가파르게 감소하는 방향인 $p_k = - \nabla f_k$로 이동 방향을 설정하는 방식이다.
>
> - 장점<br>
> : 미분값만 계산하면 되기 때문에 계산 속도가 빠르다.
> - 단점<br>
> : 수렴 속도가 느리다.
>
> **※ 이 방향이 왜 가파른 방향이고 감소하는 방향일까?**<br>
> 이를 증명하기 위해서는 Taylor's theorem(2-1)은 다음과 같이 다시 표현할 수 있다.
> 
> $$
> f(x + \alpha p) = f(x) + \alpha \nabla f(x)^Tp + \frac{1}{2} \alpha^2 p^T \nabla^2 f(x+tp)p
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
> ---
> #### Convergence rate
>
> Steepest Descent + Exact Line search = 선형적 수렴 속도
>
> 2차 함수 $f(x) = \frac{1}{2} x^T Q x - b^T x$를 예시로 생각해 볼 때, $f(x_k - \alpha \nabla f_k)$의 미분값이 0이 되는 $\alpha$를 계산해 보면 다음과 같다.
>
> $$
> \alpha_k = \frac{\nabla f_k^T \nabla f_k}{\nabla f_k^T Q \nabla f_k}
> $$
>
> 이때 $\alpha$의 값은 Rayleigh quotient에 의해 Q의 eigenvalue를 사용해 등고선이 형성된다.<br> 
> 또한 $x_{k+1} = x_k - \frac{\nabla f_k^T \nabla f_k}{\nabla f_k^T Q \nabla f_k} \nabla f_k$ 이므로 가장 가파른 방향의 기울기를 선택하다 보면 지그재그로 수렴하는 것을 알 수 있다.
>
> - 조건수<br>
> : $\rho = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}}$ 이 클수록 수렴이 매우 느려지고, 작을수록 수렴이 매우 빨라진다.
>
> ※ $\lambda$는 Q의 고유값

### 2) Newton's method

> Taylor근사를 사용해 이동 방향을 $p_k = - (\nabla^2 f_k)^{-1}\nabla f_k$로 설정하는 방식이다.
>
> - 장점<br>
> : 해 근처에서 수렴 속도가 제곱으로 빨라진다.(특히, 해 근처에서 매우 빨라짐)
> - 단점<br>
> : 해와 멀리 떨어진 곳에서는 f가 감소하는 방향이 아닐수도 있다.<br>
> : Hessian을 계산해야하기 때문에 많은 비용이 발생한다.
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
> ※ $\nabla^2 f_k \succ 0$이 아닐 경우 Newton's method는 성립되지 않는다. 이 때는 $\nabla^2 f_k^{-1}$이 존재하지 않을 수 있고, 만약 존재한다 해도 $\nabla f_k^T p_k^N < 0$ 이 성립하지 않을 수 있기 때문이다. 이러한 상황을 해결하기 위해 실제 알고리즘들은 몇가지 수정 작업을 거친다. (후술 예정)
> 
> ※ Steepest decent와 다르게 Newton's method에서는 step size를 보통 1로 설정한다. 이때, 결과가 만족스럽지 않을 경우에 step size를 줄이는 방식으로 구현한다.
>
> ---
> #### Hessian modification
>
> 위에서 보았듯이 Newton's method는 Hessian matrix($\nabla^2 f(x)$)가 positive definite여야 한다는 특징이 있었다. 하지만 우리는 이 점을 보장할 수 없기 때문에 몇가지 수정 작업을 거쳐서 구현한다.
>
> $$
> \nabla^2 f(x_k) p_k = - \nabla f(x_k)
> $$
> 
> 먼저 newton방법에서는 위의 식에 따라 방향 $p_k$를 구한다. 하지만 만약 Hessian matrix($\nabla^2 f(x)$)가 positive definite가 아니라면 이 방향이 descent direction이라는 보장을 할 수 없다.
>
> 이 문제를 해결하기 위해 Hessian과 계산 도중에 형성되는 Positive diagonal matrix나 fulll matrix를 더해서 새로운 계수 행렬을 얻어서 활용한다.
> 
> 즉 $B_k = \nabla^2 f(x_k) + E_k$를 인수분해하고, 만약 $\nabla^2 f(x_k) \succ 0$이면 $E_k = 0$ 그렇지 않으면 $B_k \succ 0$으로 만드는 $E_k$를 적절히 선택한다. 그리고 이 $B_k$를 Hessian대신에 사용하는 방식이다.
> 이때, 이 $B_k$가 다음 조건을 만족하는 선에서 modified hessian은 전역적 수렴이 보장된다.
>
> - bounded modified factorization property<br>
> : $k(B_k) = \Vert B_k \Vert \Vert B_k^{-1} \Vert \leq C, \qquad \text{some } C > 0$ 
>
> ---
> #### Theorem
>
> $f$가 두번 연속 미분 가능하고, 시작점 $x_0$의 level set $\mathcal{L} = \begin{Bmatrix} x \in \mathcal{D} : f(x) \leq f(x_0) \end{Bmatrix}$이 유계이고, 닫힌 집합이라면, bounded modified factorization을 만족한다는 가정 하에 $\lim_{k\rightarrow \inf} \nabla f(x_k) = 0$이 보장된다.
>
> ---
> #### Modification statgies
>
> - ⅰ) Eigenvalue modification<br>
> : Spectral decomposition이론에 의해 $\nabla^2 f(x_k) = Q\Lambda Q^T = \sum^n_{i=1} \lambda_k q_i q_i^T$로 분해할 수 있다. 여기서 eigenvalue가 음수인 부분($\lambda_iq_iq_i^T$)을 머신 정밀도($\mathbb{u}$)보다 큰 값중 작은 양수($\delta = \sqrt{\mathbb{u}}$)으로 교체하는 것을 말한다.<br>
> $$B_k = A + \Delta A = Q (\Lambda + diag(\tau_i))Q^T, \qquad \tau_i = \begin{cases}0 & , \lambda_i \geq \delta \\ \delta-\lambda_i & , \lambda_i < \delta\end{cases}$$
>   - 단점<br>
>     : e음수였던 eigenvalue를 작은 양수값으로 교체하면 해당 eigenvector 방향에서는 곡률이 매우 작아진다. 즉, 수정된 Hessian에서 계산된 step 방향은 이 방향에 대해 매우 긴 step을 생성할 수 있다.(미분값과 평행하기 때문) 따라서 newton's method의 장점이었던 quadratic approximation이 유효하지 않을 수 있다.
>   - example<br>
>     : $\nabla^2 f(x_k) = diag(10, 3, -1), \mathbb{u} = 10^{-16}$이라고 할 때, $B_k = diag(10, 3, 10^{-8})$이다.
>
>   - 참고<br>
>     : 위 방법 말고도 음수인 eigenvalue를 양수로 뒤집는 방식을 사용할수도 있다. 
>
> - ⅱ) Diagonal modification<br>
> : Eigenvalue modification같은 경우는 단순히 음수였던 부분만 양수로 만들어주지만, 이 방식은 $\nabla^2 f_k$의 모든 eigenvalue에 같은 값을 더해주어 minimum값이 $\delta$에 근사하도록 만드는 방식을 의미한다. 수식적으로는 다음과 같이 표현된다.<br>
> $B_k = A + \Delta A = A + \tau I, \qquad \tau = \max(0, \delta - \lambda_{\min}(A))$
> 
> ※ 이를 Software로 구현할 때는 보통 Hessian에 대해 Spectral decomposition을 직접 사용하지는 않고, 가우시안 소거법을 사용해 수정값을 선택한다.
> 
> ---
> 이외에도 오직 PSD일 경우에만 Cholesky 분해($A= L L^T$)가 가능하다는 점을 이용해서 다음과 같은 알고리즘을 만들수도 있다.
>
> - ⅲ) Cholesky with Added Multiple of the identity<br>
> : Diagonal modification에서 decomposition을 사용하는 대신에 Cholesky 분해를 시도해보는 방식이다. 즉, 특정 값 $\tau_0$에서 부터 시작하여 분해가 가능할 때 까지 $2^k \tau_0$를 반복해서 수행하면 $A + \tau I \succ 0$을 만족하는 값을 찾을 수 있다.
>
> - ⅳ) Modified cholesky factorization LDL form<br>
> (이 과정은 아직 이해하지 못했다...)
>
> - ⅴ) Modified symmetric indefinite factorization<br>
> : 4번 알고리즘에 존재하는 문제점(LDL분해 과정에서의 오류 증폭 가능성)을 해결하기 위한 방식으로 다음 성질을 이용<br>
> $$
> PAP^T = LBL^T
> \rightarrow P(A + E)P^T = L(B + F)L^T
> $$

위에서 보았다 싶이 Newton's method는 매력적인 방법이지만 계산비용이 크다는 단점이 있었다. 이에 이를 해결하기 위한 여러가지 방법이 제안 되었고 이 중 Quasi-newton's method 방법을 알아보자.

### 3) Quasi-Newton's method

Quasi-newton 방식은 Hessian $\nabla^2 f_k$을 직접 구하지 않고 $B_k$라는 근사 행렬을 사용한다. 이때 $B_k$는 각 step마다 업데이트 된다.

$$
p_k = - B_k^{-1} \nabla f_k
$$

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

※ Newton's method와 마찬가지로 해 근처에서 매우 빠른 수렴 속도를 갖는다.

### 기타

- Nonlinear conjugate gradient method<br>
: $p_k = -\nabla f(x_k) + \beta_k p_{k-1}$,<br>
($\beta_k$는 $p_k$와 $p_{k-1}$이 scalar를 유지하게 만드는 conjugate)

---
### --Step size 설정 방식--

위에서는 $\alpha$가 단순히 $f$를 감소시키는 값이라고 정의하였다. 하지만 이 정의만으로는 $f$가 최적의 값에 수렴할 수 있다고 보장할수는 없다. 그렇다고 $\min_{\alpha > 0} f(x_k + \alpha_k p_k)$의 적의 값을 찾는 것은 비용이 너무 많이든다. 즉, $f$를 충분히 감소시키는 값으로 설정해야 하고 이에 대한 정의가 필요하다.

- Wolfe conditions
- Goldstein conditions

위의 방식으로 "충분한 감소"조건에 대한 정의를 내리고 다음과 같은 방식으로 Step size를 결정하게 된다. 이때, $\alpha$를 구하기 위해 함수값을 활용할 수 있지만, 이론적으로 성립하도록 하기 위해서는 $\alpha$를 매우 좁은 구간에서 구해야 한다. 따라서 보통은 Gradient정보를 활용하여 wolfe condition이나 goldsteion condition을 통해 step size를 정의한다.

이때 이 과정은 다음과 같이 두 단계로 나누어진다.

- Bracketing phase<br>
: $\alpha$의 후보들을 순서대로 찾고, 이 중 특정 조건을 만족시키는 구간을 찾는다.

- Selection phase<br>
: 위에서 찾은 구간 내에서 보간법(Bisection/interpolation)을 통해 더 나은 $\alpha$값을 선택한다.

### @ Wolfe conditions

> #### Armijo condition
>
> $$
> f(x_k + \alpha p_k) \leq f(x_k) + c_1 \alpha \nabla f_k^T p_k \\
> c_1 \in (0, 1)
> $$
> 
> 즉, $\alpha$로 인한 실제 함수 감소량이, 기울기를 이용해 예측한 1차 근사량 감소량($\alpha \nabla f_k^Tp_k$)의 일정비율($c_1$, 보통 $10^{-4}$) 이상이여야 한다는 조건을 말한다.
>
> ![alt text](/assets/img/post/convex_optimization/armijo_condition.png)
> 
> 위 식은 이 그림과 같이 표현할 수 있다.
> - 빨간색 선 = 현재 점($x_k$)에 접하는 직선
> - 점선 = 현재 점($x$)에 접하는 직선에서 기울기를 일정 비율 줄인 직선
> - 실선 = $f(x_k + \alpha p_k)$
> 
> 위에서 $\phi(\alpha) \leq l(\alpha)$가 성립하는 step size($\alpha$)를 구하는게 목표가 된다.<br>
>
> - 한계<br>
> : 이 조건은 $\alpha$가 아주 작은 값이라면 항상 만족시킬 수 있다는 단점이 있다. 이 경우에는 최적화의 진행이 지연될 수 있기 때문이다.
>
> 이 한계를 극복하기 위해 다음과 같은 조건이 추가적으로 도입된다.
>
> ---
> #### Curvature condition
>
> $$
> \nabla f(x_k + \alpha_k p_k)^T p_k \geq c_2 \nabla_k^T p_k\\
> c_2 \in (c_1, 1)
> $$
>
> 여기서 좌변은 단순히 $\phi(\alpha)$에 대한 미분값이다. 즉, 다시말해 Step size만큼 이동한 곳에서의 기울기($\phi'(\alpha_k)$)가 현재 기울기($\phi'(0) = \nabla f_k^T p_k$)의 일정 비율($c_2$, 보통 0.9)이상 커야 한다는 조건이다.
>
> 이 조건은 움직인 이후의 기울기가 여전히 너무 음수라면(즉 경사가 급하다면) 아직 더 이동할 수 있는 가능성이 있다는 뜻이므로 $\alpha$를 더 길게 하고, 만약에 충분한 감소가 없다면 line search를 종료할 수 있게 만들어 준다.
>
> ---
> #### Lemma
>
> $f: \mathbb{R}^n \rightarrow \mathbb{R}$이 연속하고 미분 가능하고,<br>
> $p_k$가 현재 위치($x_k$)에서의 감소방향($p_k$)이고,<br>
> $[x_k+\alpha p_k \vert \alpha > 0]$에서 f가 bound이하로 떨어지지 않고,<br>
> $0 < c_1 < c_2 < 1$을 가정할 경우 wolfe condition을 만족하는 구간이 반드시 존재한다.
>
> ※ 평균값 정리 등을 이용해 증명 가능

위의 두 조건을 합친 것이 wolfe conditions라고 한다

※ Strong wolfe conditions

$$
\vert \nabla f(x_k + \alpha_k p_k)^T p_k \vert \leq c_2 \vert \nabla f_k^T p_k \vert
$$

이때, 위의 조건을 만족시키는 점 중 Stationary point(미분값이 0인 점)근처에 있지 않은 점들이 있을 수 있다. 이를 위해서 curvature condition을 수정하여 사용하는데 이를 strong wolfe conditions라고 한다.

### @ Goldstein conditions

> **Goldstein conditions**
> 
> ![alt text](/assets/img/post/convex_optimization/goldstein_condition.png)
> 
> $$
> f(x_k) + (1-c) \alpha_k \nabla f_k^T p_k \leq f(x_k + \alpha_k p_k) \leq f(x_k) + c\alpha_k \nabla f_k^T p_k \\
> 0 < c < \frac{1}{2}
> $$
>
> 두번째 조건(오른쪽)은 위에서 봤던 Armijo condition과 동일하다. 첫번째 조건은 Step size가 지나치게 작아지지 않도록 조절하는 역할을 한다.
>
> 주로 newton-type 알고리즘에 자주 사용되지만 quasi-newton알고리즘에서처럼 Hessian근사를 Positive definite로 유지해야하기 때문에 적합하지 않다.
>
> - 단점<br>
> : 그림에서 볼 수 있듯이 첫번 째 조건으로 인해 최적점이 배제될수도 있다.

### 1) Backtracking Line Search

> Wolfstein conditions에서 Curvature condition이 없을 경우 최적화의 진행이 더디게 될 수 있다고 했었다. 하지만 Backtraking 알고리즘을 사용할 경우 이러한 문제는 생략할 수 있다.
> 
> ```python
> def set_params():
>     alpha_max = 1
>     rho = randon.randrange(0, 1)
>     c = randon.randrange(0, 1)
>     return alpha_max, rho, c
> 
> alpha, rho, c = set_params()
> x = now
> p = - differential(f, x)
> 
> while f(x + alpha * p) <= f(x) - c * alpha * (p**2):
>     alpha = rho * alpha
> ```
> 
> ![](/assets/img/post/convex_optimization/backtracking_linesearch.png)
> 
> 이 그림에서 $\alpha → c, t → \alpha, \beta → \rho$라고 생각하고 보면 된다. 즉, 그냥 큰 $\alpha$에서 부터 $\rho$를 곱하면서 크기를 줄이다가 armijo condition을 만족하는 순간이 오면 그거부터 쓰자는 것이다.
> 
> ※ $\rho$를 매번 고정하지 않고  “safeguarded interpolation” 등으로 조절하여 더 효율적으로 𝛼를 축소할 수도 있다.
>
> 이 방식은 주로 newton's method에서 잘 사용되지만 quasi-newton과 conjugate method등에서는 이보다는 Wolfe조건을 더 많이 사용한다.

### 2) Interpolation

Backtracking line search를 조금 더 개선한 버전으로 가능한 적은 횟수로 $\nabla$를 계산해 효율성을 높이고자 하는 것이 목표이다. Armijo condition은 다음과 같다.

$$
\phi(\alpha) = f(x_k + \alpha p_k) \\
\phi(\alpha) \leq \phi(0) + c_1 \alpha_k \phi'(0)
$$

> 1. 초기값 확인<br>
> : 만약 초기 추정치 $\alpha_0$이 위의 condition을 만족하면 search를 마친다. 그렇지 않으면 아래의 보간법을 사용하 새로운 Step size $\alpha_1$을 구한다.
>
> 2. Quadratic Interpolation<br>
> : $\phi(0), \phi'(0), \phi(\alpha_0)$의 세 정보를 이용해 2차함수 $\phi_q(\alpha)$를 보간하여 다음과 같이 구성한다.<br>
> $$
> \phi_q(\alpha) = \frac{\phi(\alpha_0) - \phi(0) - \alpha_0 \phi'(0)}{\alpha_0^2}\alpha^2 + \phi'(0)\alpha + \phi(0)
> $$<br>
> 그리고 이 2차함수의 최소점을 구하여 새로운 Step을 구한다. $\alpha_1 = -\frac{\phi'(0)\alpha_0^2}{2[\phi(\alpha_0) - \phi(0) - \phi'(0)\alpha_0]}$
> 
> 3. Cubic Interpolation<br>
> : 만약 Quadratic Interpolation으로도 적절한 값을 구하지 못하면 $\phi(0), \phi'(0), \phi(\alpha_0), \phi(\alpha_1)$의 네 정보를 이용해 2차함수 $\phi_q(\alpha)$를 보간하여 다음과 같이 구성한다.
>
> $$
> \phi_c(\alpha) = a\alpha^3 + b\alpha^2 + \alpha \phi'(0) + phi(0) \\
> \begin{bmatrix}a \\ b\end{bmatrix} = \frac{1}{\alpha_0^2\alpha_1^2(\alpha_1 - \alpha_0)}\begin{bmatrix}\alpha_0^2 & -\alpha_1^2 \\ -\alpha_0^3 & \alpha_1^3 \end{bmatrix} \begin{bmatrix} \phi(\alpha_1) - \phi(0) - \phi'(0)\alpha_1 \\ \phi(\alpha_0) - \phi(0) - \phi'(0) \alpha_0 \end{bmatrix}
> \\
> \rightarrow \alpha_2 = \frac{-b + \sqrt{b^2 - 3a\phi'(0)}}{3a}
> $$
> 
> ※ 만약 3차보간에서도 값을 찾지 못하면 3차 보간을 구할 때 사용했던 $\phi(\alpha_1)$ 대신 이전에 구했던 $\phi(\alpha_2)$를 사용해 다시 Cubic interpolation을 반복한다. 이 과정을 armijo condition을 만족할 때 까지 반복한다.
>
> ※ 이 과정에서 $\alpha_i \approx \alpha_{i-1}$이거나, $\alpha$가 너무 작아진다면, $\alpha_i = \frac{\alpha_{i-1}}{2}$으로 재설정한다.
>
> ※ 미분값이 쉽게 계산될 수 있는 상황에서는 3차보간에서 이를 활용해 더 그럴듯한 step size를 뽑아내기도 한다.

### ※ initial step size

- newton, quasi-newton<br>
 : $\alpha_0 = 1$<br>
 (1로 잡아야 해 근처에서 빠르게 수렴하는 특성이 발휘된다.)

- steepest descent<br>
 : $\alpha_0 = \alpha_{k-1} \frac{\nabla f(x_{k-1})^T p_{k-1}}{\nabla f(x_k)^Tp_k}$<br>
 (이전 반복에서의 함수값 변화와 기울기 정보를 사용해 현재 반복에서의 $\alpha_0$을 결정하는 방식이다.)

### --Convergence--

Line search 알고리즘이 결국 $\nabla f_k = 0$ 인 안정점으로 수렴한다는 것은 Zoutendijk 정리에 의해 증명이 가능하다. 이때 수렴한다는 것이 최소점에 도달한다는 것이 아니라 $\nabla f_k = 0$ 인 점임을 유의하자.

이를 위해서 필요한 조건들은 다음과 같다.
- (a) 라인 서치가 충분한 Armijo 조건과 Curvature 조건, 그리고 Wolfe, Goldstein을 만족하는 것이 중요하다.
- (b) 탐색 방향 $p_k$가 음의 기울기를 갖고, 현재 위치에서의 기울기와 지나치게 직교하지 않음이 보장되어야 한다.
- (c) Newton/Quasi-Newton 계열에서는 Hessian 근사 $B_k$가 양의 정부호이고 조건수가 유한해야 한다.



---
---
## 4. Trust regioin 알고리즘

(pass)
