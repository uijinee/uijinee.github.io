---
title: "1. Notations"
date: 2024-03-14 22:00:00 +0900
categories: ["Math", "Linear Algebra"]
tags: ["math"]
use_math: true
---


## 1. Review of linear algebra

### 1) Notation
> - $\mathbb{R}$: 실수의 집합(Scalar)
>   - $ x \in \mathbb{R} \rarr$ x는 스칼라 
> - $\mathbb{R}^n$: n차원 실수 벡터(Vector)
>   - $ \textbf{x} \in \mathbb{R}^n \rarr$ x는 <u>Column</u> 벡터
>   - $ \textbf{x} = \begin{bmatrix} x_1,\\ x_2, \\ ..., \\ x_n\end{bmatrix} \rarr$ Matrix Notation
>   - $ \textbf{x} = \begin{pmatrix}x_1, x_2, ..., x_n \end{pmatrix} \rarr$ Coordinate Notation
> - $\mathbb{R}^{m \times n}$: Row=M, Column=N인 행렬(Matrix)
>
> ---
> | Scalar        | Vector                      | Matrix       |
> |---------------|-----------------------------|--------------|
> |  $x$ (소문자)  | $\textbf{x}$ (두꺼운 소문자) |  $X$ (대문자) |
>
> ---
> 
> | $\mathbb{S}^{n \times n}, \mathbb{S}^{n}$ | $I$ | $A^{-1}$  | $A^T$ |
> |:-----------------------------------------:|:---:|:---------:|:-----:|
> |  대칭행렬  | 단위행렬 | 역행렬| 전치행렬 |

### 2) Formula
>
> - $A\textbf{x} = \textbf{b}$
>   - $A \in \mathbb{R}^{m \times n}$
>   - $\textbf{x} \in \mathbb{R}^{n}$
>   - $\textbf{b} \in \mathbb{R}^{m}$ 
>
>> <u>행렬 연산 시 모양(Dimension)이 맞는지 항상 확인해야 한다.</u><br>
>>
>> 1. 내적과 외적
>>      - $\textbf{x, c} \in \mathbb{R^n}$
>>          - $\textbf{c}^T \textbf{x} = \textbf{c} \cdot \textbf{x}\rarr$ Scalar
>>          - $\textbf{c} \textbf{x}^T = \textbf{c} \times \textbf{x} \rarr$ Rank-1 Matrix (nxn)
>>
>> 2. Quadratic Form
>>      - $Q \in \mathbb{S}^n, \textbf{x} \in \mathbb{R}^n$
>>          - $\textbf{x}^T Q \textbf{x} \rarr$ Scalar
>>          - $\textbf{x}^T Q \textbf{x} \rarr \textbf{x}$의 Quadratic Form(2차함수)
>>          - $f(x)=\textbf{x}^T Q \textbf{x} \rarr ax^2 \rarr$ 아래로 볼록(Convex Function)
> 
> ---
> - ${AA^{-1} = A^{-1}A = I}$
> 
> - $(AB)^{-1}=B^{-1}A^{-1}$
> 
> - $(AB)^{T}=B^{T}A^{T}$
> 
> - $(A+B)^{T}=A^{T}+B^{T}$
> 
> - $(A^{-1})^{T} = (A^T)^{-1} = A^{-T}$
>
