---
title: "1. Time Complexity"
date: 2022-04-11 22:00:00 +0900
categories: ["Computer Science", "Algorithm"]
tags: ["cs", "algorithm"]
use_math: true
---

---
# 시간복잡도

## 1. 복잡도 카테고리

### 1) 시간복잡도란?
시간 복잡도란 어떤 알고리즘의 시행 속도를 표현하는 수식을 의미한다.
이 표현 수식의 종류에는 표현 목적에 따라 다음과 같이 총 5가지가 있다

> **Tight Bound를 구할 때**
> - `Big O Notation` : 점근적 상한 *(Tight Upper Bound)*
> - `Big Omega Notation`: 점근적 하한 *(Tight Lower Bound)*
>
> ---
> **Loose Bound를 구할 때**
> - `Little O Notation` : Loose Upper Bound
> - `Little Omega Notation` : Loose Lower Bound

*(참고)<br>
(`Theta Notation` : Big O와 Big Omega가 같을 때 Theta Notation으로 표현할 수 있다.)*


### 2) Big-O Notation 종류
Big-O Notation의 종류는 다음과 같다.

> #### Polynomial Time Complexity
>
> | Complexity | | 예시 |
> |:---:| --- |:---:|
> | $O(1)$ | Constant Complexity | Hesh |
> | $O(\text{loglog} n)$ | | |
> | $O(\text{log} n)$ | Logarithmic Complexity | 이진탐색 |
> | $O(n)$| Linear Complexity | Scan |
> | $O(n \text{log} n)$ | | 병합정렬<br> Quick정렬 |
> | $O(n^2)$ | Quadratic Complexity | 삽입정렬<br> 선택정렬 |
> | $O(n^3)$ | Cubic Complexity | 행렬곱 | 
>
> ---
> #### Exponential Time Complexity
>
> | Complexity | | 예시 |
> |:---:| --- |:---:|
> | $O(2^n)$ | Exponential Complexity | Knapsack Problem<br> Fibonacci<br> Hanoi |
>
> ---
> &#8251; 참고: 분류 방법
> 
> $O( f(n) )$ 이라고 할 때 $f(n)$에서 제일 영향이 큰 항을 고르고 그 계수들을 버리면 위의 카테고리 중 한개를 얻을 수 있다.


### 3) 증명

위 분류 방법은 상한선(Tight Upper Bound)이 같은 알고리즘들로 나눈 것이다.
즉, 해당 알고리즘들은 최악의 경우에도 해당 시간복잡도보다는 많이 걸리지 않는다는 뜻이다.

> 위의 분류방법은 다음과 같은 순서로 증명할 수 있다.<br>
> $g(n)$의 시간이 걸리는 함수가 있고 $f(n)$은 위 카테고리중 하나라고 하자.
>
> | 1. $g(n) \leq cf(n)$이라는 부등식을 세운다. |
> | 2. 부등식을 만족할 만한 임의의 c를 선택한다. (최대한 Tight하게) |
> | 3. C에 대해 부등식을 만족하는 n의 범위를 구한다. |
> | 4. $n_0 < n$인 모든 n에 대해 부등식을 만족하는 $n_0$를 선택한다. (최대한 작은 값으로) |
> | 5. $n_0 < n$인 모든 n에 대해 $g(n) \leq cf(n)$을 만족하므로 $g(n)$은 $f(n)$에 포함된다. |
>
> 즉, 위 과정을 만족하는 $n_0$와 $c$를 찾는 과정을 통해 증명이 가능하다
>
> ---
> &#8251; **주의할 점**
> - $n_0$를 구할 때 모든 실수에 대해 만족하는 것인지 확인해야 한다.
> - $O(n)$에는 $O(\text{log}n)$과 같은 복잡도들도 포함된다.
>

---
## 2. 구하는 방법
먼저 입력의 크기에 따라 변하는 부분은 변수 n을 가지고 표현하고 나머지 부분은 통틀어서 1로 표현한다.<br>
그 다음 해당 식을 위의 방법대로 해석하면 된다.

### 1) 반복문의 시간복잡도

```c++
void example(int n){
    int v[10];
    for(int i=0; i<10; i++)
        v[i] = -1;
    for(int i=0; i<n; i++)
        v[i] = (i>n)?i:n;
    for(int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            v[i] = (i>j)?i:j;
}
```

> #### 1. 데이터의 크기에 따라 변하는 for문에 대해서만 계산한다.
>
> 위의 example에서 각 For문은 다음과 같이 표현할 수 있다.
> 
> - $O(1)$<br>
>   ```c++
>   for (int i=0; i<10; i++)
>       v[i] = -1;
>   ```
> 
> - $O(n)$<br>
>   ```c++
>   for (int i=0; i<n; i++)
>       v[i] = (i>n)?i:n;
>   ```
>
> - $O(n^2)$<br>
>   ```c++
>   for(int i=0; i<n; i++)
>       for (int j=0; j<n; j++)
>           v[i] = (i>j)?i:j;
>   ```
>
> $\Rightarrow$ 즉, $O( T(n) ) = O( n^2 + n + 1) = O(n^2)$



### 2) 재귀함수의 시간복잡도

```c++
int BinarySearch(int target, int a[], int left, int right){
    if(left <= right){
        int mid = (left + right) / 2;
        if (target == a[mid])
            return mid;
        else if (target < a[mid])
            return BinarySearch(target, a, left, mid-1);
        else if (target > a[mid])
            return BinarySearch(target, a, mid+1, right);
    }
}
```

> #### 1. 재귀함수를 점화식으로 표현한다.
> 
> 위의 BinarySearch는 다음과 같이 표현할 수 있다.
>
> - $T(1)$<br>
>   ```c++
>   if(left <= right){
>       int mid = (left + right) / 2;
>   ```
>
> - $T(\frac{n}{2})$<br>
>   ```c++
>   if (target == a[mid])
>       return mid;
>   else if (target < a[mid])
>       return BinarySearch(target, a, left, mid-1);
>   else if (target > a[mid])
>       return BinarySearch(target, a, mid+1, right);
>   ```
>
> &#8251;참고: if문 이므로 밑의 코드는 하나만 실행된다.
>
> $\Rightarrow$ 즉,$T(n) = T(n/2) + 1$
>
> *(만약 case문이었다면 if문이 둘 다 실행되어 점화식이 바뀐다.)*
>
> ---
> #### 2. 시간복잡도를 구한다.
>
> 점화식으로부터 직접 유도하는 방식도 있지만 다음 이론도 사용 가능하다.
>
> | | Master Theorem | Advanced Master Theorem | 
> | --- | --- | --- |
> | | $T(n) = aT(\frac{n}{b}) + f(n)$ | $T(n) = aT(\frac{n}{b}) + n^k \text{log}^p(n)$ |
> | 조건 | ⅰ. $a \geq 1$ 이다.<br> ⅱ. $b > 1$ 이다. <br> ⅲ. $f(n)$은 $n \geq 0$일 때 항상 양함수이다.<br> ⅳ. $af(\frac{n}{b}) \leq cf(n)$을 만족하는 c가 존재한다.<br> $\quad (c<1)$ | . $a \geq 1$ 이다.<br> ⅱ. $b > 1$ 이다. <br> ⅲ. $f(n)$은 $n \geq 0$일 때 항상 양함수이다.<br> ⅳ. $af(\frac{n}{b}) \leq cf(n)$을 만족하는 c가 존재한다.<br> $\quad (c<1)$ |
> | 시간복잡도 | $h(n) = n^{\text{log}_ba}$라고 할 때,<br> $$T(n) = \begin{cases} O(f(n)), \quad f(n) > h(n) \\ O(f(n) \times \text{log}n), \quad f(n) = h(n) \\ O(h(n)), \quad f(n) < h(n) \end{cases}$$ | ⅰ. $a > b^k$일 때<br> $\quad T(n) = O(n^{log_ba})$<br><br> ⅱ. $a = b^k$일 때<br> $$\quad T(n) = \begin{cases} O(n^{\text{log}_ba} \text{log}^{p+1} n), \quad p > -1 \\ O(n^{\text{log}_ba} \text{log log} n), \quad p = -1 \\ O(n^{\text{log}_b a}), \quad p < -1 \end{cases}$$<br><br> ⅲ. $a < b^k$일 때<br> $$\quad T(n) = \begin{cases} n^k \text{log}^p n, \quad p \geq 0 \\ n^k, \quad p<0 \end{cases}$$ |