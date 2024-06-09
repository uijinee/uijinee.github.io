---
title: "6. Object Detection(One-Stage)"
date: 2024-03-28 22:00:00 +0900
categories: ["Artificial Intelligence", "Deep Learning(Basic)"]
tags: ["cnn", "detection"]
use_math: true
---

# Object Detection

## 1. One-Stage Detection Model

### 1) YOLO

![alt text](/assets/img/post/deeplearning_basic/yolo.png)

> #### Purpose
>
> 1. 앞선 모델들은 ROI를 위해 Box Regression과정을 한번 더 수행했다.<br>
>   $\rightarrow$ Grid Cell
>
> _(논문 제목 그대로 You Only Look Once, Box Regression과정은 한번이면 족하다.)_
> _(제목부터 자극적이다...)_
>
> ---
> #### 동작과정
>
> ![alt text](/assets/img/post/deeplearning_basic/yolo_procedure.png)
>
> 1. Grid
>   : 이미지를 $7 \times 7$크기의 Grid로 나눈다.<br>
>
> 2. Classification<br>
>   : 각 Grid에 대해 Classification을 수행한다.<br>
>    _(Background도 이 Class중 1개에 포함)_
>
> 3. **Base Box 생성**<br>
>   : 각 Grid에 대해 "B"개의 Base Box들을 생성한다.<br>
>   _(이 Base Box가 Anchor Box의 역할을 함)_
>
> 4. 각 Base Box별로 Regression을 수행한다.<br>
>   Output $\rightarrow$ (x, y, h, w, Confidence)<br>
>   _(grid cell기준 x, grid cell기준 y, image기준 h, image기준 w, objectness)_
>
> $\therefore 7 \times 7\times (5 \times B + C)$개의 Bounding Box 생성 
> 

### 2) DETR

![alt text](/assets/img/post/deeplearning_basic/detr.png)

> #### Purpose
> 
> 1. 기존의 Anchor Box를 사용하는 방식은 다음과 같다.<br>
>   Positive Sample 결정 $\rightarrow$ NMS (or other Hand Craft Post Processing) $\rightarrow$ ROI결정<br>즉, Ground Truth에 대해 Prediction이 Many-to-one 관계이다.<br><br>
>   $\Rightarrow$ Anchor Box을 없애 Hand craft한 Post Processing이 없는 One-to-one model을 만듦
>   
> ---
> #### 동작과정
>
> | 과정 | | | |
> | --- | --- | --- | --- |
> | 1. Backbone | ![alt text](/assets/img/post/deeplearning_basic/detr_procedure(1).png) | $(C, H, W)$<br> $\rightarrow (D, HW)$ | ⅰ. Flatten<br>　: Transformer의 Query로 주기 위한 평탄화<br>ⅱ. Positional Encoding<br>　: Patch의 위치정보를 위한 Encoding |
> | 2. Transformer<br>(Encoder) | ![alt text](/assets/img/post/deeplearning_basic/detr_procedure(2).png) | $(D, HW)$<br> $\rightarrow (D, HW)$ | Encoder는 **Multi-head Self Attention**을 통해<br> Key와 Value를 생성 |
> | 3. Transformer<br>(Decoder) | ![alt text](/assets/img/post/deeplearning_basic/detr_procedure(3).png) | $(D, HW), N$<br> $\rightarrow N$ | ⅰ. N개의 Object Query(초기값=0) 생성하고<br>　**Multi-head Self Attention**<br>ⅱ. Encoder의 Key-Value와 함께<br>　**Multi-head Cross Attention**<br><br> ☆ **Query는 Object Prompt & Learnable**<br>　 |
> | 4. Feed Foward | ![alt text](/assets/img/post/deeplearning_basic/detr_procedure(4).png) | $N \rightarrow N$ | ⅰ. N개의 Query에 대해 모두 예측<br>　_(Background의 경우 예측 X)_<br> ⅱ. Set to Set 학습을 위한<br>　**Bipartite Matching**|
> 
> _(Self Attention: Query, Key, Value가 같은 Feature Map에서 생성되는 것)_<br>
> _(Cross Attention: Query는 생성 Key, Value를 Encoder에서 가져오는 것)_
>
> ---
> #### Bipartite Matching
>
> ![alt text](/assets/img/post/deeplearning_basic/bipartite_matching.png)
>
> DETR은 Output이 Parallel하게 출력된다. 즉, Box Regression결과를 한번에 알게 된다.
>
> 따라서 각 Box가 어떤 Ground Truth와 Matching되어야 하는지 알 수 없다.
>
> 이때, Brute Force하게 최적의 조건을 탐색하면 시간이 지나치게 많이 걸린다.<br>
> 이를 위해 사용하는 것이 Bipartite Matching(이분 매칭)으로, 각 Bounding Box가 어떤 Ground Truth와 연결되어야 최적의 상태가 되는지 알아내는 알고리즘이다.
>
> $$
> \hat{\sigma} = \text{arg}\,\min \limits_{\sigma \in \Omega_N} \sum \limits_i^N \mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)}) 
> $$
> 
> - 최적의 쌍($\hat{\sigma}$)은 두 노드를 잇는 연결선 중($\sigma \in \Omega_N$) 유사도($\mathcal{L}_{match}$)를 최소화 하는 쌍
> 
> $$
> \mathcal{L}_{match} = -\mathbb{1}_{\{c_i \neq \emptyset\}} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \emptyset\}} \mathcal{L}_{box}(b_i, \hat{b}_{\sigma(i)})
> $$
> 
> 　
> $$
> \mathcal{L}_{Hungarian}(y, \hat{y}) = \sum \limits_{i=1}^N [-log(\hat{p}_{\hat{\sigma}(i)}(c_i)) + \mathbb{1}_{\{c_i \neq \emptyset\}} \mathcal{L}_{box}(b_i, \hat{b}_{\hat{\sigma}}(i))] 
> $$
>
> 　

-- todo: SSD, RetinaNet, Efficient Det, M2Det, CorNerNet --