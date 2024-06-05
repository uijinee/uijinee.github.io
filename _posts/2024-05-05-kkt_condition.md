---
title: "4. KKT Condition"
date: 2024-05-05 22:00:00 +0900
categories: ["Math", "Convex Optimization"]
tags: ["math"]
use_math: true
---

## 1. Duality

### 1) Lagrangian

> #### Lagrangian Function
>
> $$
> L(\mathbf{x}, \lambda, \nu) = f_0(\mathbf{x}) + \sum \limits_{i=1}^m \lambda_i f_i(\mathbf{x}) + \sum \limits_{i=1}^p \nu_i h_i(\mathbf{x})
> $$
> 
> ---
> #### Lagrangian Dual Function
>
> $$
> g(\lambda, \nu) = \inf \limits_{\mathbf{x} \in \mathbb{R}^n} L(\mathbf{x}, \lambda, \nu)
> $$
>
> - lower bound property<br>
> $g(\lambda, \nu) \leq p^* = f(\mathbf{x}^*) \qquad , if \;\; \lambda \geq 0$ 
> 
> ---
> #### Lagrangian Dual Problem
>
> $$
> maximize \quad g(\lambda, \nu)\\
> subject \; to \quad \lambda \geq 0
> $$

## Examples: 

1. least norm solution of linear equations
2. linear programming
3. equality constrained norm minimization
4. Entropy Maximization
5. Tow way Partitioning
