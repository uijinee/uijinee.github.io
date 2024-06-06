---
title: "7. Machine Learning"
date: 2024-05-01 22:00:00 +0900
categories: ["Artificial Intelligence", "AI"]
tags: ["ai"]
use_math: true
---

# Machine Learning

Machine Learning에는 크게 2가지 종류가 존재한다.

1. Deductive Learning<br>
    규칙들을 분석해 새로운 규칙을 찾아내는 것

2. Inductive Learning<br>
    데이터에 기반해 규칙을 찾아내는 것
    - Supervised Learning(지도학습)<br>
        : label(정답)이 있는 학습과정<br>
        $\rightarrow$ _Classification(분류), Regression(함수값 예측)_
    - Self-Supervised Learning<br>
        : label(정답)을 임의로 만들어 Embedding(representation)을 학습하는 것<br>
        (후에 학습 된 representation를 활용하여 모델이 좋은 성능을 가지게 할 수 있음)<br>
        $\rightarrow$ _Self-Prediction, Contrastive Learning_
    - Unsupervised Learning(비지도 학습)<br>
        : label(정답)이 없는 학습과정<br>
        $\rightarrow$ _Clustering_
    - Reinforcement Learning<br>
        : label(정답)이 없고 보상을 통해 행동을 학습하는 것

---
## 1. Hypotheses

### 1) Hypothesis

> Machine Learning의 목표는 미래의 Data(Test Data)에 대해서 잘 작동하는 함수를 찾는 것이다.
> 이를 위해서 다음과 같은 개념이 도입된다.
> 
> - Target Function<br>
>   : Input에 대해 우리의 Task에 알맞은 Output을 내는 목표함수<br>
>   (즉, Target Function은 "Unknown"이다.)
> 
> - Hypothesis Function<br>
>   : Train Data를 관찰한 결과 Target Function일 것이라고 가정한 함수
> 
> - Generalization<br>
>   : Train에서 학습한 모델이 Test Data에서도 잘 동작하도록 하는 것
>
>> 즉, 우리의 목표는 Generalization이 잘 된 Hypothesis Function을 찾는 것이다.
>
> ---
> #### Bias & Variance
>
> ![alt text](/assets/img/post/machine_learning/overfitting_underfitting.png)
>
> - Bias<br>
>   : Hypothesis Space와 실제 Target Function과의 거리를 의미한다.
> - Variance<br>
>   : Hypothesis Space의 범위를 의미한다.<br> 
>   범위가 너무 크면 Target Function뿐만 아니라 다른 Noise들도 배우기 때문에 문제가 발생한다.
>
> Error는 Bias와 Variance의 합으로 표현할 수 있다.

### 2) Hypothesis Spaces

> 가능한 모든 Hypothesis Function이 속한 공간을 의미한다.<br>
>
>> 즉, 이 Hypothesis Space에서 Target Function과 가장 유사한 Hypothesis Function을 찾아야 한다.
>
> ---
> #### 주의점
>
> - Under Fitting<br>
>   : train data으로 학습할 때, Target Function을 찾지 못해 발생하는 상황이다.<br>
>   $\rightarrow \quad \because$ Bias 
>   
> - Over Fitting<br>
>   : Target Function은 찾았지만 Generalization이 되지 않은 상황이다.<br>
>   $\rightarrow \quad \because$ Variance
> 
> 이러한 Under Fitting과 Over Fitting현상을 막기위해서는 적절한 Complexity를 갖는 Model을 사용하는 것이 중요하다.
>
> &#8251; _Ockham's Razor_<br>
> $\quad$ Ockham's Razor는 필요 이상의 복잡한 가설(모델)을 사용하지 말자는 일종의 철학을 말한다.
>
> ---
> #### Probable Hypothesis
>
> 그럼 이제 이 Hypothesis Space에서, 어떻게 Best Hypothesis Function을 찾을 수 있을까?<br>
> 이 문제를 풀기 위해 Bayes' Rule을 적용해보자
>
> 우리는 데이터가 주어졌고 가설 h가 있을 때, h는 간단하게 정의되어 있을 확률이 높으므로 h가 작을 때 $P(h)$는 커진다고 정의하자.
>
> $h^* = \text{arg}\max \limits_{h \in \mathcal{H}} P(h \| data)$<br>
> $\quad\; = \text{arg}\max \limits_{h \in \mathcal{H}} \frac{P(data \| h)P(h)}{P(data)}$<br>
> $\quad\; = \text{arg}\max \limits_{h \in \mathcal{H}} P(data \| h)P(h)$
>
> 즉, 최적의 가설 $h^*$는 데이터를 잘 설명하면서, 가장 간단한 가설이 된다.

---
## 2. Model

### 1) Loss Function

> #### Error Rate
>
> 예측값과 정답이 다른 비율을 의미한다.
>
> $E(h) = \frac{1}{N} \sum \limits_{i=1}^N [[h(x) \neq y]]$
> 
> ---
> #### Loss Function
>
> Loss Function이란 최적의 상황에서의 Utility랑 현재 상황에서 Uitlity의 차이를 의미한다.<br>
> $L(x, y, \hat{y}) = Utility(x, y) - Utility(x, \hat{y})$<br>
>
> 참고로 이때, $L(a, b) \neq L(b, a)$임을 주의하자.
>
>
> ⅰ. **0/1 Loss**<br>
> $$
> \qquad L_{0/1}(y, \hat{y}) = 
> \begin{cases}
> 0, \quad (y=\hat{y}) \\
> 1, \quad (otherwise)
> \end{cases}
> $$
>
> ⅱ. **Absolute-Value Loss**<br>
> $$
> \qquad L_1(y, \hat{y}) = \vert y-\hat{y} \vert
> $$ 
> 
> ⅰ. **Squared-Error Loss**<br>
> $$
> \qquad L_2(y, \hat{y}) = (y-\hat{y})^2
> $$ 
> 
>
> ---
> #### Generalization Loss
>
> 위의 Loss Function을 정의하고 난 후, 우리는 Utility의 기댓값인 Generalization Loss를 정의할 수 있다.<br>
> 
> $$
> \text{GenLoss}_L(h) = \sum \limits_{(x, y) \in \epsilon} P(x, y)L(y, h(x))
> $$
>
> 이때, $P(x, y)$를 알 수 없으므로 이를 $\frac{1}{N}$으로 두고 계산하기도 한다.<br>
> 즉, 인공지능은 다음의 식을 풀게 된다.
>
> $$
> h^* = \text{arg}\min \limits_{h \in \mathcal{H}} \text{EmpLoss}_L(h) = \text{arg}\min \limits_{h \in \mathcal{H}} \frac{1}{N} \sum \limits_{(x, y) \in \epsilon} L(y, h(x))
> $$
>
> ---
> &#8251; 우리는 Data가 iid(independent identically distributed)라고 가정한다.<br>
> $$
> \quad P(E_j) = P(E_j \| E_{j-1}, E_{j-2}, ...) \\
> \quad P(E_j) = P(E_{j+1}) = P(E_{j+2})
> $$
> 

### 2) Model Selection

![alt text](/assets/img/post/machine_learning/valid.png)

> #### Cross-Validation
>
> ![alt text](/assets/img/post/machine_learning/cross_validation.png)
>
> Train과 Validation을 동시에 하면서, Validation error가 가장 작을 때의 모델을 고르는 것
>
> ```python
> def model_selection(learner, examples, k):
>   err = []
>   train, test = examples.split()
>   for size in range(1, 1e9):
>       err[size] = cross_validation(learner, size, train, k)
>       if err[size] > err[size-1]:
>           best = min(err[:size-1])
>           h = learner(best, train)
>           return h, error_rate(h, test)
>
> def cross_validation(learner, size, examples, k):
>   N = len(examples) 
>   err = 0
>   for i in range(k):
>       valid = examples[(i-1)*N/k : i*N/k]
>       train = examples - valid
>       h = learner(size, train)
>       err = err + error_rate(h, valid)
>   return err / k
>
> ```
> 
> ---
> #### Hyperparameter Tuning
>
> | 1. Hand-Tuning | 직관을 사용해 조정<br>　 |
> | 2. Grid Search | 몇몇 Hyperparameter들에 대해 가능한 값들을 설정하고<br> 이 값들의 모든 조합에 대해 최적의 값을 찾는 방식<br> ex. `param_grid = {'n_estimators': [100, 150, 200],`<br>$\qquad\qquad\quad\qquad$ `'max_depth' = [None, 6, 9, 12], }`<br>　 |
> | 3. Random Search | random하게 Hyperparameter들을 정해 최적의 값을 찾는 방식<br>　|
> | 4. Bayesian Optimization | validation data에서 각 Hyperparameter에 대한 성능을 평가하고<br> 이를 통해 Hyperparamter($x$)와 성능($y$)간의 함수 $y=f(x)$를 추정하는 방식<br>　 |
> | 5. Population-based Training | ⅰ. Sequential Search: Bayesian Optimization<br> ⅱ. Parallel Search: Random Search<br> 두 종류의 방법은 장단점이 있으므로 두 방법을 연결하여 사용해보자<br>　 |

### 3) Regularization

> 실제 우리가 구한 가설($\hat{h}^*$)과 실제 결과인 target function($f$)가 차이나는 이유는 다음과 같다.
>
> - Training Data의 Variance
> - f에 존재하는 Noise
> - H의 Complexity의 한계로 인한 구현 불가($f \notin \mathcal{H} $)
>
> 이 문제를 해결하기 위해 Regularization이 필요하다.<br>
> Regularization은 가설공간 $\mathcal{H}$에 추가적인 제약을 걸어 Variance를 줄여주는 역할을 한다.
>
> $$
> Cost(h) = EmpLoss_{L, E}(h) + \lambda \text{Complexity}(h) \\
> \hat{h}^* = \text{arg}\min \limits_{h \in \mathcal{H}} Cost(h)
> $$
>
> ---
> &#8251; 만약 Complexity를 L1-Norm으로 한다면 Feature Selection 효과를 얻을 수 있다.