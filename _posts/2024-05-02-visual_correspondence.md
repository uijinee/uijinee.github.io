---
title: "1. Visual Correspondence"
date: 2024-05-02 22:00:00 +0900
categories: ["Artificial Intelligence", "Computer Vision"]
tags: ["visual correspondence", "vision"]
use_math: true
---

# Image Matching

## 1. Classical Method

Image Matching의 Classical한 Pipeline은 다음과 같다.

| Feature Detect | $\rightarrow\qquad$ Feature Descriptor | $\rightarrow\qquad$ Matching |
|:---:|:---:|:---:|
| $$\begin{bmatrix} \text{'Salient'}\\ \text{Repeatable'} \end{bmatrix}$$한 Point 찾기 | 점의 특징을 수학적으로 표현 | $N \times M$개의 Matching Candidate 비교 |

&#8251; Salient: 두드러진<br>
&#8251; 8-Point Algorithm: 8개의 Matching Point를 통해 두 이미지의 기하학적 관계를 추정하는 알고리즘

### 1) SIFT

![alt text](/assets/img/post/computer_vision/sift.png)

> #### Step1: Detection
> 
> SIFT는 Blobs Detector로, Feature Detect시에 Blob한 지점들을 찾는다.<br>
> (&#8251; Harris Corner: Corner Detector)<br>
> 또 추가적으로 Scale Invarient하게 동작할 수 있도록 설계한다.
> 
> Blob을 찾을 수 있는 방법에는 다음 2가지 방법이 있다.
>
> | LoG(Laplacian of Gaussian) | DoG(Difference of Gaussian) |
> |:---:| --- |
> | ![alt text](/assets/img/post/computer_vision/log.png)<br> ![alt text](/assets/img/post/computer_vision/log_result.png) | ![alt text](/assets/img/post/computer_vision/gaussian.png) <br> ![alt text](/assets/img/post/computer_vision/dog.png) |
> | Gaussian Filter에 Laplacian을 적용한 Filter,<br> BandPass Filter, Blob한 점에서 큰 반응을 보임<br> _($\sigma$를 변화시키며 가장 큰 반응이 오는 값을 찾음)_ | Gaussian Pyramid에서 같은 Octave에 있는<br> 이미지 2개를 뺀 값으로 LoG에 근사할 수 있다. |
>
> LoG이미지나, DoG이미지들을 <u>필터의 크기나 이미지의 크기를 조절해가며 Pyramid 구조로 쌓음</u>으로써 **Scale Invarient**하게 동작하도록 설계할 수 있다.
>
> 여기서 유의해야 할 점이 LoG Pyramid를 사용할 경우 다음과 같은 문제가 발생한다는 점이다<br>
> - Computational Cost가 크다
> - 미분 시 Noise가 증폭되는데 Log는 미분을 두번한다.
>
> 즉, 위와 같은 문제들을 방지하기 위해 SIFT에서는 DoG Pyramid를 사용한다.
>
> $$
> \Downarrow
> $$
> 
> | 3D NMS | KeyPoint Localization |
> | --- | --- |
> | ![alt text](/assets/img/post/computer_vision/3dnms.png) | $D(\mathbf{x}) = D + \frac{\partial D^T}{\partial \mathbf{x}} \mathbf{x} + \frac{1}{2} \mathbf{x}^T \frac{\partial^2 D}{\partial \mathbf{x}^2}\mathbf{x}$<br>$D(\mathbf{x})' = 0$<br>_(DoG 이미지($D$)의 2차 근사의 극값)_<br> $\qquad \Downarrow$ <br> ![alt text](/assets/img/post/computer_vision/keypoint_localization.png) |
> | 위에서 구한 DoG Scale Space에서 상하좌우<br> 28개 점과 비교하여 가장 큰 점을<br> Corner후보군으로 선택한다. | 3D NMS를 통해 고른 점은 Corner뿐만 <br> 아니라 Edge도 포함되어 있다.<br> SIFT에서는 이를 걸러줄 방법으로<br> DoG 이미지의 Hessian Matrix를 이용한다. |
>
> $$
> \Downarrow
> $$ 
> 
> 지금까지 ⅰ) Dog를 통해 특징점 후보군 추출, ⅱ) 3D NMS + KeyPoint Localization으로 특징점의 위치와 크기를 확정하였고<br> 
> 이제 ⅲ) 특징점의 방향을 할당해 줄 차례이다.
>
> | ![alt text](/assets/img/post/computer_vision/orientation_assignment.png) | **Orientation Assignment**<br><br> 특징점의 위치에서 Scale범위 내의 모든 점들에 대해<br> 다수가 가리키는 방향으로 Keypoint의 방향을 설정한다. |
> | ![alt text](/assets/img/post/computer_vision/keypoint_orientation(2).png) | 이때, Dominant한 방향이 여러개일 경우<br> 각각의 방향을 가지는 Blob을 여러개 설정한다. |
> 
> ---
> #### Step2: Description
> 
> Step1에서 KeyPoint를 구했고, 이제 Matching을 위해 서로 다른 KeyPoint를 식별/구분할 수 있는 Finger Print를 만들어야 한다.<br>
> 이를 Descriptor라고 한다.
>
> | ![alt text](/assets/img/post/computer_vision/descriptor_gradient.png) | Image에서 Keypoint별로 $4 \times 4$개의 Window로 나눈다.<br> 그리고 이 Window를 다시 $4 \times 4$개의 Sub-Window에<br> 할당한다. (총 $16 \times 16$으로 나뉨)<br> 마지막으로 각 구역별로 Image Pixel값에 기반해<br> Gradient를 계산한다.  | 
> | ![alt text](/assets/img/post/computer_vision/descriptor_gaussian.png) | 이 Gradient들을 Noise에 Robust하게 작동하도록<br>하기 위해 Gaussian Filter를 적용해준다. |
> | ![alt text](/assets/img/post/computer_vision/descriptor_histogram.png) | 이제 Window별로 할당된 모든 구역에 대해 <br> 8개의 방향을 갖는 Orientation Histogram을 그린다.<br>_(Window당 $8 \times 4 \times 4 = 128$차원의 Vector가 생긴다.)_ |
> | ![alt text](/assets/img/post/computer_vision/descriptor_result.png) | 완성 |

---
## 2. Deep Learning

### 1) Matching(Metric Learning)

Metric Learning에서는 Hinge Loss를 주로 사용하여 Matching여부의 결정경계를 찾는다.

$$
Loss(y, \hat{y}) = \min \limits_w \begin{Bmatrix} \frac{\lambda}{2} \Vert w \Vert_2 + \sum \limits_{i=1}^N \text{max}(0, 1-y_i\hat{y}) \end{Bmatrix}
$$

&#8251; Metric Learning: 입력 이미지간의 유사도(거리)를 학습하는 모델

> #### [Paper1: DeepCompare](https://arxiv.org/pdf/1504.03641)
>
> &#8251; Network: Descriptor 역할<br>
> &#8251; Decision Layer: Matching 역할
> 
> | Architecture | |
> | --- | --- |
> | **2-Channel Model**<br> ![alt text](/assets/img/post/computer_vision/metric_2-channel.png) | 2개의 Patch를 Channel방향으로 Concatenate하여 입력함<br><br>**문제점**<br> Patch Pair간의 중복된 부분을 Pair-Wise하게 계산하기 때문에<br>Descriptor의 재사용이 불가능하다.<br>_(ex. \[P1, P2]에서 사용한 Descriptor는 \[P1, P3]에서 사용 불가 )_ <br>$\Rightarrow$ Computational Cost가 큼<br> |
> | **Siamese Model**<br> ![alt text](/assets/img/post/computer_vision/metric_siamese.png) | 1개의 Patch($P_1$)에 대해 Descriptor를 추출하고, 이 가중치를<br> 그대로 사용하여 다른 Patch($P_2$)의 Descriptor를 추출<br><br> $\Rightarrow$ $P_1$과 ($P_3, P_4, P_5...$)를 비교할 때,<br>$\quad$ 추가적인 계산이 필요 없다. |
> | **2-Channel 2-Stream Model**<br>![alt text](/assets/img/post/computer_vision/metric_2channel_2stream.png) | 2개의 독립적인 Stream을 통해 Patch를 비교하는 방식<br> (2-Channel모델 2개를 합쳐놓은 모델)<br><br> ⅰ. Stream 1: 두 Patch의 중심부를 비교하는 Stream<br>ⅱ. Stream 2: 두 Patch의 주변부를 비교하는 Stream <br><br> $\Rightarrow$ 다양한 Receptive Field에서 비교할 수 있음 | 
>
> $\Rightarrow$ Decision Network가 여전히 필요하다는 단점
>
> ---
> #### [Paper2: DeepDesc](https://openaccess.thecvf.com/content_iccv_2015/papers/Simo-Serra_Discriminative_Learning_of_ICCV_2015_paper.pdf)
>
> | Architecture | |
> | --- | --- |
> | ![alt text](/assets/img/post/computer_vision/deepdesc.png) | 별도의 Decision Network없이 Siamese Network에서 추출한<br> Descriptor Vector들의 L2-Norm을 출력<br><br> **Pairwise Hinge Loss**<br> $$l(\mathbf{x}_1, \mathbf{x}_2) = \begin{cases} \Vert D(\mathbf{x}_1) - D(\mathbf{x}_2) \Vert_2, \quad p_1 = p_2 \\ \text{max}(0, C-\Vert D(\mathbf{x}_1) - D(\mathbf{x}_2) \Vert_2), \quad p_1 \neq p_2\end{cases}$$ |
> 
> #### Hard Negative Mining
> 
> **Hard Negative**란 실제로는 Negative인데 Positive라고 잘못 예측하기 쉬운 데이터를 말한다. <br>
> 반면에 **Easy Negative**란 실제로도 Negative이고 예측도 Negative라고 예측하기 쉬운 데이터를 말한다.<br>
>
> 주로 False Positive Sample이 Hard Negative데이터가 되는데, 이 이유는 보통 Positive에 해당하는 것을 학습하는 것을 목표로 하기 때문에 잘못 예측한 False Negative Sample은 잘 고려하지 않기 때문이다.<br>
>
> Hard Negative Mining은 이런 Sample들을 추출해 데이터셋에 포함시켜 학습하는 방법이다.
> 
> ---
> #### [Paper3: Triplet Learning]
> 
>![alt text](/assets/img/post/computer_vision/triplet_learning.png)
> 

### 2) Orientations

18분

> #### [Paper1: LIFT]()
