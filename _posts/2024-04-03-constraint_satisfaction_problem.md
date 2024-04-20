---
title: "4. CSP"
date: 2024-04-03 22:00:00 +0900
categories: ["Artificial Intelligence", "Machine Learning"]
tags: ["deeplearning", "machine learning", "csp", "constraint satisfaction"]
use_math: true
---

# Constraint Satisfaction Problem

## 1. Background

### 1) 목표

> Domain Specific한 Heuristic Function이 아닌 General 한 Heuristic Function으로 문제를 해결하는 것

### 2) 구성요소

> | $\mathcal{X}$ | 변수 | $\begin{Bmatrix}X_1 & X_2 & ... & X_n \end{Bmatrix}$ | 
> | $\mathcal{D}$ | Domain<br> 변수마다 각각의 Domain이 존재함 | $\begin{Bmatrix}D_1 & D_2 & ... & D_n \end{Bmatrix}$ |
> | $\mathcal{C}$ | 변수들이 갖는 Constraint<br>　ⅰ. Ornary Constraint: 변수가 1개인 constraint<br>　ⅱ. Binary Constraint: 변수가 2개인 Constraint | <범위, 관계>의 형태<br><br>ex. $\langle(X_1, X_2), \begin{Bmatrix}(3, 1), (3, 2), (2, 1) \end{Bmatrix} \rangle$<br>$\langle(X_1, X_2), X_1 > X_2 \rangle$<br> |

### 3) Constraint hypergraph

![alt text](/assets/img/post/machine_learning/map_coloring.png)

> 
> | ![alt text](/assets/img/post/machine_learning/cryptarithmetic.png) | 1. 이와 같은 문제가 있다. <br><br> 이 문제는 다음과 같이 표현할 수 있다.<br> $\langle O+O = R+10\cdot C_1 \rangle$<br>$\langle C_1 + W + W = U + 10 \cdot C_2 \rangle$<br>$\langle C_2 + T + T = O + 10 \cdot C_3 \rangle$<br>$\langle C_3 = F \rangle$<br> |
> | ![alt text](/assets/img/post/machine_learning/constraint_graph.png) | 2. 이 문제는 다시 Constraint Graph로 그릴 수 있다. |
> 

## 2. Constraint Propagation

위와 같이 문제를 모델링한 후에, 변수 하나에 임의의 값을 설정해보자.<br>
그러면 Constraint에 의해 다른 변수들이 가질 수 있는 값에 영향이 간다.

이때, 이 영향은 Constraint에 직접적으로 참여하는 변수 뿐만 아니라 간접적으로 참여하는 변수도 포함이다. 이를 Constraint Propagation이라 한다.

### 1) Arc Consistency Enforcing

> - **Arc consistent**<br>
>   : Consistent Propagation을 위해서는 변수 $X$의 정의역의 모든값이 이 Constraint에 참여하게 해야한다. 이를 **Arc consistent**한 상태라고 정의한다.
>
> ---
> #### Algorithm
>
> 