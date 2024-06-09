---
title: "1. Feedfoward Neural Network"
date: 2024-03-13 22:00:00 +0900
categories: ["Artificial Intelligence", "Deep Learning(Basic)"]
tags: ["ai"]
use_math: true
---

# Feedfoward Neural Network

## 1. 구조

---
## 2. Train 

### 1) Loss Function

> #### 종류
> 
> 대표적인 Loss Function들에는 다음이 있다.
> 
> | Squared Error | Negative Log Likelihood | Cross Entropy |
> | --- | --- | --- |
> | $L(\mathbf{y}, h_w(\mathbf{x})) = \sum \limits_{i=1}^n (y_i - \hat{y}_i)^2$ | $L(\mathbf{y}, h_w(\mathbf{x})) = - log (P_w(\mathbf{y} \vert \mathbf{x}))$ | $$\mathbb{H}(P, Q) = - \mathbb{E}_{\mathbf{z \sim P(\mathbf{z})}}[log(Q(\mathbf{z}))] \\ \qquad \quad \;\; = - \int P(\mathbf{z}) log(Q(\mathbf{z})) \\ \mathbb{H}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum \limits_i y_i log(\hat{y}_i)$$ |
> | 예측값과 정답사이의 L2-Norm | 입출력 관계를 얼마나 잘 설명하는지 | P의 분포와 Q의 분포사이의 거리 |
>
> ---
> #### Minimization
>
> - 직접 계산<br>
> : $\mathbf{W}^* = \text{arg}\min \limits_\mathbf{W} \sum \limits_{\mathbf{x, y} \in \mathcal{D}} L(\mathbf{y}, \mathbf{h}_\mathbf{W}(\mathbf{x}))$
> 
> - Gradient Descent<br>
> : $\mathbf{W} = \mathbf{W} - \alpha \nabla \text{Loss}(\mathbf{W})$

### 2) Backpropagation

그렇다면 각 Layer마다 Loss의 Gradient는 어떻게 구할 수 있을까??

> | ![alt text](/assets/img/post/deeplearning_basic/backpropa_example.png)<br> $\text{Loss}(\mathbf{W}) = L_2(y, h_\mathbf{W}(\mathbf{x})) = (y-\hat{y})^2$<br>$\text{act}(y) = \text{Sigmoid}(y) = \frac{1}{1+e^{-y}}$ | ⅰ. $w_{3, 5}$의 Gradient 계산 $\rightarrow \frac{\partial \text{Loss}(\mathbf{W})}{\partial w_{3, 5}}$<br>$\quad$ ※ $y_5 = \text{bias} + x_3w_{3, 5} + x_4w_{4, 5}$<br>$\quad$ ※ $\hat{y} = x_5 = \text{act}(y_5)$<br><br> $$\frac{\partial \text{Loss}(\mathbf{W})}{\partial w_{3, 5}} \\ = \frac{\partial (y-\hat{y})^2}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial w_{3, 5}} \\ = \frac{\partial (y-\hat{y})^2}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial y_5} \times \frac{\partial y_5}{\partial w_{3, 5}} \\ = -2(y-\hat{y}) \times \hat{y}' \times x_3 \\ = \;\vartriangle_5 \times x_3$$<br><br>ⅱ. $w_{1, 3}$의 Gradient계산 $\rightarrow \frac{\partial Loss(W)}{\partial w_{1, 3}}$<br>$\quad$※ $y_5 = \text{bias} + x_3w_{3, 5} + x_4w_{4, 5}$<br>$\quad$※ $y_3 = \text{bias} +  x_1 w_{1, 3} + x_2 w_{2, 3}$<br>$\quad$※ $\hat{y} = x_5 = \text{act}(y_5)$<br>$\quad$※ $x_3 = \text{act}(y_3)$ <br><br> $$\frac{\partial \text{Loss}(\mathbf{W})}{\partial w_{1, 3}} \\ = \frac{\partial (y-\hat{y})^2}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial w_{1, 3}} \\ = \frac{\partial (y-\hat{y})^2}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial y_5} \times \frac{\partial y_5}{\partial x_3} \times \frac{\partial x_3}{\partial y_3} \times \frac{\partial y_3}{w_{1, 3}} \\ = -2(y-\hat{y}) \times \hat{y}' \times w_{3, 5} \times \hat{y}' \times x_1 \\ =\; \vartriangle_3 \times x_1$$ |
>
> 
>> 위의 계산과정을 보면 알 수 있듯이 층마다 Gradient가 계속 곱해지는 것을 알 수 있다.<br>
>> 
>> 이때, Sigmoid같은 경우는 미분값이 최대 0.25기 때문에 깊은 층일수록 Gradient의 변화가 매우 작아질 수 밖에 없음을 알 수 있다.<br>
>> 이러한 현상을 **Gradient Vanishing현상**이라고 한다.
>>
>> **방지 대책**
>> - ReLU<br>
>>   : Gradient가 1이 되도록 하여 깊은 층에서도 작아지지 않게 하는 방법
>>   - 단점: 층이 깊어질 수록 입력값이 0이되는 경우가 많아져 Dying ReLU현상이 발생한다.
>> - Skip Connection
>>   : 별도의 Connection을 통해 깊은 층에서도 Gradient를 유지하는 방법
> 
> ---
> #### Gradient Flow
>
> | ![alt text](/assets/img/post/deeplearning_basic/gradient_graph.png) | ![alt text](/assets/img/post/deeplearning_basic/gradientflow.png) |
>
> Backpropagation을 Graph로 정리해보면 위와 같다.

### 3) Optimization

SGD

> #### Algorithm
>
> ![alt text](/assets/img/post/deeplearning_basic/sgd_sudo.png)
>
> ---
> #### 장점
>
> - Stochasticity를 유지할 수 있다. (Local Minimum 탈출 가능)
> - Parallelism하게 동작할 수 있다.


---
## 3. Generalization

복잡한 가설에는 Penalty를 주는 방법

### 1) Weight Decay

$\text{arg} \min \limits_\mathbf{W}(\sum \limits_{(x, y)}L(\mathbf{y}, h_\mathbf{W}(\mathbf{x}))+ \lambda \sum \limits_{i, j} W_{i, j}^2)$

> ![alt text](/assets/img/post/deeplearning_basic/weightdecay_map.png)
> 
> Weight Decay는 MAP로써도 해석할 수 있다.

### 2) Drop Out

> 몇몇 노드를 일정 확률로 비활성화하여 각 노드들이 더 많은 정보를 학습할 수 있도록 하는 것
>
> - 매 학습마다 다른 구조의 모델이 학습되는데 때문에 Ensemble(Bagging)의 효과를 얻을 수 있다는 해석도 존재
>
> &#8352; Drop Connect: Node가 아닌 Weight를 Drop하는 방법