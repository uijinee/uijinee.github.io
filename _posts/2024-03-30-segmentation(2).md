---
title: "8. Segmentation(Instance, Panoptic)"
date: 2024-03-30 22:00:00 +0900
categories: ["Artificial Intelligence", "Deep Learning(Basic)"]
tags: ["cnn", "segmentation"]
use_math: true
---

# Segmentation

## 1. Background

### 1) Segmentation의 종류

> Segmentation의 Task를 구분하는 가장 중요한 "Things"와 "Stuff"이다.
> - Things: 일반적으로 볼 수 있는 물체들을 의미한다.
> - Stuff: 하늘, 땅, 길, 숲 등과 같은 무정형의 지역을 의미한다.
> 
> | Semantic Segmentation | Instance Segmentation | Panoptic Segmentation |
> | --- | --- | --- |
> | ![alt text](/assets/img/post/deeplearning_basic/semantic_segmentation.png) | ![alt text](/assets/img/post/deeplearning_basic/instance_segmentation.png) | ![alt text](/assets/img/post/deeplearning_basic/panoptic_segmentation.png) |
> | Things와 Stuff를 모두 탐지<br>But Instance 구분 X  | Things만 탐지<br> But Instance 구분 O | Things와 Stuff를 모두 탐지<br> And Instance 구분 O |
> | 1. Supervised Learning<br>2. Unsupervised Learning  | 1. Bottom-Up 방식<br>2. Top-Down 방식 | |
>
> ---
> #### Instance Segmentation
>
> 1. Bottom-Up Approches<br>
>   : 이미지의 개별 픽셀을 감지하는 것부터 시작해, 픽셀들을 그룹화하며 객체를 탐지하는 방식
>       - _ex. WT, InstanceCut, SGN, DeeperLab, Panoptic-DeepLab_
>
> 2. Top-Down Approches<br>
>   : 이미지의 Bounding Box를 찾고 Binary Segmentation을 진진행하는 방식
>       - _ex. Mask R-CNN, PANet, HTC, YOLACT_
>

### 2) Panoptic Quality

> Panoptic Segmentation은 Stuff까지 함께 Prediction하기 때문에 조금 다른 성능 평가 지표를 사용한다.
>
> ![alt text](/assets/img/post/deeplearning_basic/panoptic_quality.png)
>
> $$
> PQ = \frac{\sum \limits_{(p, q) \in TP} IoU(p, q)}{\|TP\| + \frac{1}{2} \|FP\| + \frac{1}{2} \|FN\|}
> $$ 


---
## 2. Instance Segmentation

### 1) Mask R-CNN

![alt text](/assets/img/post/deeplearning_basic/mask_rcnn.png)

> #### **Purpose**
>
> | Faster R-CNN | Mask R-CNN |
> | --- | --- |
> | ![alt text](/assets/img/post/deeplearning_basic/faster_rcnn.png) | ![alt text](/assets/img/post/deeplearning_basic/mask_rcnn.png) |
>
> 1. Top-Down Approches<br>
>   $\rightarrow$ Faster RCNN에 Mask Branch(FCN) 추가
>
> 2. Faster R-CNN의 RoI Pooling에서 발생하는 **Quantization Error**는 Detection Problem에서는 중요하지 않았다.<br>
>   하지만 Segmentation에서는 Pixel단위의 정확도가 필요하다<br>
>   $\rightarrow$ RoI Pooling대신 RoI Align 사용
> 
> ---
> #### 동작과정
>
> 1. **FPN** _(Feature Pyramid Network)_<br>
>   : 다양한 크기의 Object들을 감지하기 위해 FPN Backbone을 사용한다.
>
> 2. **RPN** _(Region Proposal)_<br>
>   : Feature Map에서 물체가 있을만한 곳(RoI)을 고른다.<br>
>   _(그 후, Objectness Score가 높은 K개를 골라 NMS를 수행해 준 후 다음 Layer로 전송)_
>
> 3. RoI Align
>   : RPN에서 찾은 Bounding Box후보들을 FPN의 Feature Map에 Projection한다.<br>
>   이때, RoI Pooling을 사용하는 기존의 방식은 정확도가 떨어지기 때문에 RoI Align이라는 방법을 사용한다. 
>  
> | **4. MLP** | **4. FCN** |
> | 　물체의 Bounding Box를 추정하기 위한 Layer<br><br><br>　Box Regression, Classification을 수행 |　Binary Segmentation을 위한 Layer<br> 　가장 기초적인 방법인 FCN 사용<br><br> $\rightarrow K$개(Class 개수)의 Feature Map을 출력 |
> 
> 　　　　　　　　　　　_(두 과정은 Parallel하게 동작한다.)_
> 
> 　**5. Multi-task Loss Function**<br>
> 　　: $L = L_{cls} + L_{box} + L_{mask}$ 를 통해 학습<br>
> 　　_(이때 $L_{mask}$는 Binary Cross Entropy Loss이다.)_
>
> ---
> #### RoI Align
>
> | **RoI Projection**<br> ![](/assets/img/post/deeplearning_basic/roi_projection.png) | RoI Pooling을 위해서 Bbox들을<br> Feature Mapdml Size에 맞게 Resize한다.<br><br>이때, 딱 나누어 떨어지지 않으면 반올림한다.<br> _ex. 920/16 = 57.5_ $\rightarrow$ 58 |
> | **RoI Pooling**<br> ![alt text](/assets/img/post/deeplearning_basic/roi_pooling_problem.png)| 즉, 위의 RoI Projection에서의 반올림은<br> RoI Pooling에서 다음과 같은 Quantization<br> 오류를 발생시킨다. |
>
> | **RoI Align 과정** | |
> | --- | --- |
> | ![alt text](/assets/img/post/deeplearning_basic/roialign.png) | RoI Pooling과 같이 Resize할 때, 반올림을 고려하지 않는다|
> | ![alt text](/assets/img/post/deeplearning_basic/roialign(2).png) | RoI를 $2 \times 2$ Bin으로 나눈다. _(예시)_ |
> | ![alt text](/assets/img/post/deeplearning_basic/roialign(3).png) | ⅰ. 각각의 Bin에 Sample Point 4개씩 생성<br>　_(논문에서는 각 cell의 3등분하는 점으로 잡았음)_<br>　_(논문에 따르면 Point의 위치나 개수는 중요하지 않음)_<br><br>ⅱ. 각 Sample Point마다 셀의 가로/세로 길이에 대해 <br>　Bilinear Interpolation으로 위치 계산<br> |
> | ![alt text](/assets/img/post/deeplearning_basic/roialign(4).png) | 각 Bin마다 Sample Point들을 Avg Pooling(Max Pooling)  |


---
## 3. Panoptic Segmentation

### 1) DETR for panoptic segmentation

![alt text](/assets/img/post/deeplearning_basic/detr_panoptic.png)

DETR논문에서는 Object Detection문제를 해결하며 Panoptic Segmentation에 대한 활용법도 제안하였다.

> #### Purpose
>
> Mask R-CNN과 마찬가지로 Object Detection Model인 DETR에서 Branch를 하나 추가해서 Segmentation도 풀 수 있게 만들어 보았다.
>
> ---
> #### 동작과정
>
> 1. **Encode Image** (as a Backbone)<br>
>   $\rightarrow$ DETR의 Encoder에서 이미지의 Feature Map을 뽑아온다.
>
> | **2. Classification** | **2. Segmentation** |
> | ![alt text](/assets/img/post/deeplearning_basic/detr_classification.png) | ![alt text](/assets/img/post/deeplearning_basic/detr_segmentation.png) |
> | Encoding된 Feature Map을 (Key-Value)로 사용하고<br>Things and Stuff Query와 Multi Head Attention | Encoding된 Feature Map을 (Key-Value)로 사용하고<br> Bounding Box와 Multi Head Cross Attention |
> 
> 3. FPN-Style CNN (ex. Resnet)<br>
>   : Attention Map을 다시 Up Sampling하기 위해 FPN과 같은 모델을 사용한다.
>
> 4. Pixel Wise Argmax<br>
>   : N(Query 개수)개의 Mask에 대해 Pixel Wise Argmax를 하여 Segmentation한다. 

### 2) MaskFormer

![alt text](/assets/img/post/deeplearning_basic/maskformer.png)

_(SAM:Segment Anithing Model과 같이 읽어보면 좋다고 함)_

> #### Purpose
>
> ![alt text](/assets/img/post/deeplearning_basic/perpixel_maskclassification.png)
> 
> 일반적으로 Segmentation은 Pixel별로 분류하는 것으로 알려져있다.<br>
> 하지만 이 논문의 저자는 DETR로부터 Segmentation은 Pixel별 분류가 아닌 Mask 분류라고 관점을 다르게 생각한 것 같다.
>
> 즉, Pixel별 분류를 할때는 하나의 Loss를 사용하여 Class를 예측했지만<br>
> DETR과 같이 Transformer구조가 나오면서, N개의 Mask(Bbox)를 Parallel하게 생성할 수 있게 되었고,<br>
> 이는 우리가 더이상 Pixel별 분류를 할 필요 없는 Mask Classification Task로 본다는 뜻인 것 같다. (맞나요...?)
>
> _(MaskFormer는 위와 같은 생각을 갖고, DETR+Panoptic을 조금 더 깔끔하게 만든 모델이다.)_
> 