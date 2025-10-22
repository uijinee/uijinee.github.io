---
title: "1. Basic Algorithm & Ancient Cipher"
date: 2025-09-18 12:00:00 +0900
categories: ["Math", "Cryptography"]
tags: ["cryptography"]
use_math: true
---

# Basic Algorithm

## 1. Modulo 연산

### 1) 정의

$\frac{A}{B} = Q$와 나머지 $R$을 가질 때, $A \bmod B = R$이다.

- (합동) $A \equiv B \pmod C$<br>
: 모듈로 연산 $C$ 에서 같은 값을 갖게 되는 수들($A, B$)을 모듈로 $C$에 대한 합동 관계에 있다고 한다.

- (덧셈의 역원) $A + B \equiv 0 \pmod m$<br>
: Modulo $m$에서 두 수의 합이 0과 합동일 때, 두 수는 modulo $m$에서 덧셈 역원이라고 한다.

- (곱셈의 역원) $A \times B \equiv 1 \pmod m$<br>
: Modulo $m$에서 두 수의 곱이 1과 합동일 때, 두 수는 modulo $m$에서 곱셈 역원이라고 한다.


### 2) 성질

> #### Modulo 성질
>
> - (덧셈): $(A + B) \bmod C = (A \bmod C + B \bmod C) \bmod C$
> - (곱셈): $(A \times B) \bmod C = (A \bmod C \times B \bmod C) \bmod C$
> - (제곱): $A^B \bmod C = ((A \bmod C)^B) \bmod C$
>
> ---
> #### Modulo 합동 성질
>
> - 전달성<br>
>   (가정) $a \equiv b \pmod m$ 이고 $b \equiv c \pmod m$<br>
>   (결과) $a \equiv c \pmod m$
>
> - 곱셈<br>
>   (가정) $a \equiv b \pmod m$<br>
>   (결과) $ac \equiv bc \pmod m$
> 
> - 나눗셈<br>
>   (가정) $ac \equiv bc \pmod m$ 이고 $\gcd(c, m) = 1$<br>
>   (결과) $a \equiv b \pmod m$
> 
> - CRT<br>
>   (가정) $a \equiv b \pmod p$ 이고 $a \equiv b \pmod q$ 이고 $\gcd(p, q) = 1$<br>
>   (결과) $a \equiv b \pmod pq$


## 2. 기타 공식들

### 1) Euclidean Algorithm(유클리드 호제법)

$$
GCD(a, b) = GCD(b, r), \quad \text{where } a>b, a \equiv r \pmod b
$$

> #### 최대 공약수 알고리즘
> 
> $GCD(252, 198)$<br>
> $= GCD(198, 54)$<br>
> $= GCD(54, 36)$<br>
> $= GCD(36, 18)$<br>
> $= 18$
>
> ---
> 기타 표기
> 
> - $Z_m = m-1$<br>
> (e.g. $Z_{26} = 25$)
> 
> - $Z_m^\ast = \[a \| a \in Z_m \text{ and } \gcd(a, m) = 1 \]$<br>,
> (e.g. $Z_{7} = [1, 3, 5]$)

### 2) Extended Euclidean Algorithm

$ax + by = \gcd(a, b)$를 만족하는 x, y가 존재하며, 이는 유클리드 호제법의 과정을 역으로 따라가면 찾을 수 있다.

**Example**

a = 252와 b = 198일 때 $ax + by = \gcd(a, b)$를 만족하는 x, y를 찾아라.

① $252 = 198 \times 1 + 54$<br>
② $198 = 54 \times 3 + 36$<br>
③ $54 = 36 \times 1 + 18$<br>
④ $36 = 18 \times 2 + 0$<br>
⑤ $54 - 36 \times 1=18$ ...(from ③)<br>
⑥ $54 - (198 - 54 \times 3) = 18$ ...(from ②)<br>
⑦ $4 \times 54 – 198 = 18$<br>
⑧ $4 \times (252 - 198) - 198 = 18$ ...(from ①)<br>
⑨ $4 \times 252 - 5 \times 198 = 18$ 

> #### 역원 찾기 알고리즘
>
> 곱셈에 대한 역원은 이 Extended Euclidean Algorithm을 사용하면 쉽게 찾을 수 있다.
>
> **Example**
>
> (문제) 31과 $\bmod 105$에서 역원인 수를 구하여라.
>
> $31 a + 105 b = \gcd(31, 105) = 1$을 만족하는 x, y를 찾아보자.
>
> $-44 \times 31 + 13 \times 105 = 1$ 
>
>> 이때, $(13 \times 105) \bmod 105 = 0$ 이므로,<br>
>> $\bmod 105$에서 13과 곱셈에 대해 역원인 수는 $-44(\equiv 61 \bmod 105)$이다.

---
# Ancient Cipher

