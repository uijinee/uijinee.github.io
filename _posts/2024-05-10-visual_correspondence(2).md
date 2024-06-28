---
title: "2. Visual Correspondence(2)"
date: 2024-05-02 22:00:00 +0900
categories: ["Artificial Intelligence", "Computer Vision"]
tags: ["visual correspondence", "vision"]
use_math: true
---

# Dense Correspondence

모든 점들을 활용해 Pixel간의 매칭을 수행하는 것<br>
ex. Flow예측

즉, 모든 점들을 활용해야 하기 때문에 Sparse Correspondence처럼 특별한 Feature Detection과정이 필요없다.<br>
따라서 다음으로 구성할 수 있다.
- Feature Descriptor
- Regularization<br>
: 모든 점들을 사용하는 만큼 Regularization방법이 중요해진다.
하지만 Noise가 많아지기 때문에 Regularization방법이 중요해진다.

(before: Sparse Correspondence: 특정 점들을 사용해 이미지간의 매칭을 수행하는 것)

예를들어 이미지1과 이미지2가 있다고 하자<br>
이 때 Dense Correspondence는 이미지1과 이미지2의 Pixel Level Correspondence를 다음과 같은 Energe Function을 사용하여 풀 수 있다.

![alt text](/assets/img/post/computer_vision/densecorrespondence_energefunc.png)

[참고](https://hygenie-studynote.tistory.com/53)


## 1. Stereo Matching

2개의 카메라를 사용하여 3D Depth Map을 추측하는 알고리즘

### 1) Epipolar Geometry

![alt text](/assets/img/post/computer_vision/epipolar_geometry.png)

>  한 물체를 두 카메라로 관찰할 때, 이미지 위의 점 $x, x'$에서 보인다고 하자.
>
> - Epipolar Line<br>
> : 양 카메라의 원점을 이은 점과 이미지가 겹치는 부분을 $e, e'$라고 할 때, $\overline{PE}, \overline{O'E'}$에 해당한다.
>
> - Epipolar Plane<br>
> : 검정색으로 색칠된 평면
> 
> ---
> #### Rectification
> 
> | $l' = E^Tp$<br>$p^TEp' = 0$ | **ⅰ. Essential Matrix**<br> $\quad$: Intrinsic Parameter를 고려하지 않는<br> $\quad$ 변환행렬($E$)을 구한다.<br>$\;$ |
> | ![alt text](/assets/img/post/computer_vision/fundamental_matrix.png) | **ⅱ. Fundamental Matrix**<br> $\quad$ Intrinsic Parameter를 고려하여 변환행렬을 구한다.<br> $\quad$ 이를 통해 Epipolar Line을 구할 수 있다.<br><br>※ 8-Point Algorithm<br> $\quad$: 8개의 Pair Point를 활용해 $F$를 계산하는 알고리즘<br> $\;$|
> | ![alt text](/assets/img/post/computer_vision/rectification.png) | **ⅲ. Rectification**<br> $\quad$: Epipolar line들을 평행하게 만드는 변환행렬 $H$를<br> $\quad$ 찾는 것<br> $\quad \Rightarrow$ Epipolar line이 같은 높이에 있게 되어 가로방향의<br> $\quad$ 좌표차이를 구하는 1D Search Problem이 된다. |

### 2) Stereo Matching

Stereo Matching이란, 이제 Rectification이 완료된 두 이미지에서 1D Search를 하는 과정이라고 정의할 수 있다.

※ Disparity: 1D Search시에 대응되는 두 점 사이의 거리

![alt text](/assets/img/post/computer_vision/streomatching(2).png)<br>
![alt text](/assets/img/post/computer_vision/streomatching.png)

가장 기본적인 구조는 위와 같다.<br>
즉, Window를 만들어 움직이면서 Similarity를 측정하고 이를 Maximize하는 점을 찾으면 된다.

그렇다면 이 Similarity를 어떻게 측정할 수 있을까?<br>
이 방법에 따라 Stereo Matching의 종류가 나뉜다.

> #### Classical method
>
> | ![alt text](/assets/img/post/computer_vision/stereomatching_classic.png) | ※ SAD: Matrix간의 L1-Norm<br> ※ SSD: Matrix간의 L2-Norm | 
> 
> $\Rightarrow$ **BUT** Repetitve Pattern를 갖거나 Reflective/Transparent Surface, Occlusion된 물체에 대해 성능이 떨어지는 단점이 있다.<br> 또한 Noise나 얇은 물체 등을 못잡는 문제가 발생함 
>
> ---
> #### Deep Learning
> 
> [Paper1: MC-CNN](https://arxiv.org/pdf/1510.05970)
>
> | Architecture | Absract |
> | --- | --- |
> | ![alt text](/assets/img/post/computer_vision/mccnn_architecture.png) | 기본적으로 먼저 Siamese Network를 사용하여<br> 두 Patch의 Fature를 추출<br><br>**ⅰ. MC-CNN-fst**<br> $\quad$ - 빠르게 Similarity Score를 추출하는 방법 <br>**ⅱ. MC-CNN-acrt**<br> $\quad$ - 느리지만 정확하게 Similarity Score를 추출 |

---
## 2. Mothion Estimation(Optical Flow)

Stereo Matching은 1D Search Problem이었다면 Optical Flow는 2D Search Problem이다.

### 1) [FlowNet](https://openaccess.thecvf.com/content_iccv_2015/papers/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.pdf) & [DispNet](https://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16/paper-MIFDB16.pdf)

![alt text](/assets/img/post/computer_vision/flownet.png)

> - shared network로 featuremap 추출
> - 한 Pixel에 대해 다른 Patch의 2D Window와의 Correlation Volume을 추출
> - 이 Pixel은 다시 Convolution을 통해 모양을 조절하여 Corrrelation Volume과 Concatenate
> - Convolution Layer로 구성된 Encoder와 Decoder를 거쳐 Optical Flow 예측 (like unet)
> 
> ---
> #### Refinement
>
> ![alt text](/assets/img/post/computer_vision/flownet_refinement.png)
>
> - Decoder는 위와 같은 구조로 설계되었고 학습시에는 Ground Truth를 Resize하여 Low Level에서도 학습을 진행<br> _(Multi-Task Learning)_


### RAFT

![alt text](/assets/img/post/computer_vision/raft.png)

### GMFlow

![alt text](/assets/img/post/computer_vision/gmflow.png)