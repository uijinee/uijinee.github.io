---
title: "2. Convex Function"
date: 2024-03-16 22:00:00 +0900
categories: ["Math", "Convex Optimization"]
tags: ["math"]
use_math: true
---

# Convex Function

### 1) 정의

Convex

$f$의 Domain이 Convex Set이고 $0 \leq \theta \leq 1$일 때, 모든 $\mathbf{x}_1 \mathbf{x}_2$에 대해<br>
$f(\theta \mathbf{x}_1 + (1-\theta) \mathbf{x}_2 \leq \theta f(\mathbf{x}_1) + (1-\theta)f(\mathbf{x}_2$

- Concave: $-f$

Strictly Convex

$f$의 Domain이 Convex Set이고 $0 < \theta < 1$일 때, 모든 $\mathbf{x}_1 \mathbf{x}_2$에 대해<br>
$f(\theta \mathbf{x}_1 + (1-\theta) \mathbf{x}_2 < \theta f(\mathbf{x}_1) + (1-\theta)f(\mathbf{x}_2$


## 2. 예시

### 1) $\mathbb{R} \rightarrow \mathbb{R}$

> | Function | 수식 | Convex Function | Concave Function |
> | -------- | ---- |:---------------:|:----------------:|
> | Affine   | $ax + b$ | O           | O                |
> | Exponential | $e^{ax}$ | O         | X                |
> | Powers | $x^{\alpha}$  | $\alpha \leq 0 \quad \alpha \geq 1$ | $0 \leq \alpha \leq 1$ |
> | Powers of absolute value | $\|x\|^p$ | $p \geq 1$ | |
> | Negative Entropy | $x log x$ | $x>0$ | X |
> | Logarithm | $log x$ | X | $x > 0$ |

### 2) $\mathbb{R}^n \rightarrow \mathbb{R}$

> 
>
>
>

### 3) $\mathbb{R}^{m \times n} \rightarrow \mathbb{R}$