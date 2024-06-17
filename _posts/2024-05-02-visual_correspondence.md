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
| $$\begin{bmatrix} \text{'Salient(두드러진)'}\\ \text{Repeatable'} \end{bmatrix}$$한 Point 찾기 | 점의 특징을 수학적으로 표현 | $N \times M$개의 Matching Candidate 비교 |

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
> | Gaussian Filter에 Laplacian을 적용한 Filter,<br> BandPass Filter로써 Blob한 점에서 큰 반응을 보인다.<br> _($\sigma를 변화시키며 가장 큰 반응이 오는 값을 찾음)_ | Gaussian Pyramid에서 같은 Octave에 있는<br> 이미지 2개를 뺀 값으로 LoG에 근사된 값을 가진다. |
>
> 위의 방법 LoG이미지나, DoG이미지들을 필터의 크기나 이미지의 크기를 조절해가며 Pyramid 구조로 쌓음으로써 Scale Invarient하게 동작하도록 설계할 수 있다.
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
> | ![alt text](/assets/img/post/computer_vision/descriptor_gradient.png) | Image에서 Keypoint별로 $16 \times 16$개의 구역으로 나눈다.<br> 그리고 이 구역을 각각 $4 \times 4$개의 Window에 할당한다.<br> 마지막으로 각 구역별로 Gradient를 계산한다.  | 
> | ![alt text](/assets/img/post/computer_vision/descriptor_gaussian.png) | 이 Gradient들을 Normalize해주기 위해<br> Gaussian Filter를 사용하여 크기를 조절한다. |
> | ![alt text](/assets/img/post/computer_vision/descriptor_histogram.png) | 이제 Window별로 할당된 모든 구역에 대해 <br> 8개의 방향을 갖는 Orientation Histogram을 그린다.<br>_(Window당 $8 \times 4 \times 4 = 128$차원의 Vector가 생긴다.)_ |
> | ![alt text](/assets/img/post/computer_vision/descriptor_result.png) | 완성 |

---
## 2. Deep Learning

### 1) Descriptor

> #### DeepCompare

> #### DeepDesc

### 2) Orientations

### 3) Image Matching