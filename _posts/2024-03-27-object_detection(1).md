---
title: "3. Object Detection(One-Stage)"
date: 2024-03-27 22:00:00 +0900
categories: ["Artificial Intelligence", "Deep Learning(Basic)"]
tags: ["cnn", "detection"]
use_math: true
---

# Object Detection

## 1. BackGround

### 1) 이미지 분류모델의 종류
![alt text](/assets/img/post/deeplearning_basic/classification_type.png)

> | | Classification | Object Detection | Segmentation |
> | --- | --- | --- | --- |
> | | ![alt text](/assets/img/post/deeplearning_basic/classification.png) | ![alt text](/assets/img/post/deeplearning_basic/detection.png) | ![alt text](/assets/img/post/deeplearning_basic/segmentation.png) |
> | 정의 | 이미지를 판단하는 모델 | Bounding Box를 통해<br> 객체를 탐지하는 모델 | Classification을 <br><br> Pixel단위로 수행하는 모델 |
> | Output | class번호 | bbox, class | segmentation map |
> | 종류 | | 1. One-Stage Detector<br> 2. Two-Stage Detector| 1. Semantic Segmentation<br> _(Class단위 분류)_<br><br>2. Instance Segmentation<br> _(객체단위 Class분류)_<br><br> 3. Panoptic Segmentation<br> _(Semantic + Instance)_ |
>

### 2) 용어

> #### Confusion Matrix
>
> ![alt text](/assets/img/post/deeplearning_basic/confusion_matrix.png)
>
> 모델이 얼마나 잘 예측했는지 평가하기 위해 각 Case별로 나누어 표시하는 것
>
> - $Precision=\frac{TP}{TP+FP}$<br>
>   : 모델이 True라고 분류한 모델중 실제 True인 것의 비율
>
> - $Recall=\frac{TP}{TP+FN}$<br>
>   : 실제 True인 것 중에서 모델이 True라고 예측한 것의 비율
>
> ---
> #### PR Curve
>
> ![alt text](/assets/img/post/deeplearning_basic/pr_curve.png)
>
> 매 예측에 대해 Precision과 Recall의 변화를 계산하여 그래프를 그리는 것
>
> ---
> #### AP(Average Precision)
>
> ![alt text](/assets/img/post/deeplearning_basic/average_precision.png)
>
> 
하나의 Class에 대해 PR Curve를 구했을 때, 이 PR Curve에서의 아래면적의 크기
>
- **mAP(mean Average Precision)**
: 모든 Class에 대해 AP를 구했을 때 이 AP의 평균
>
---
#### IOU(Intersection Over Union)
>
> ![alt text](/assets/img/post/deeplearning_basic/iou.png)
>
> Classification은 맞고 틀리고에 대해서 명확히 판별할 수 있었지만, Detection에서는 이것을 단정지을 수 없다.
>
> 이를 위해 IOU를 사용하는데,<br>
> 예를들어 "IOU 60"의 경우 전체 면적중에서 Bounding Box가 겹치는 부분이 60%이상인 것들만 정답으로 취급한다.<br>
> *(마찬가지로 `mAP 60`은 `IOU 60`을 기준으로 mAP를 측정한 것이다.)*
>

### 3) Selective Search

![alt text](/assets/img/post/deeplearning_basic/selective_search.png)


> ROI(Region of Interest)를 추출하는 전통적인 방법으로 다음과 같은 과정을 통해 물체가 있을법한 영역의 후보군을 추출하게 된다.
>
> - 이미지의 색갈, 질감, 모양등을 활용해 무수히 많은 작은 영역으로 나눈다.
> - 이 영역들에서 겹치는 부분이 많은 영역들을 점차 통합해 나간다.
>
> Selective Search는 정해진 알고리즘이 있어 작동원리를 이해하기 쉽다는 장점이 있었지만, 현재는 이렇게 정해져 있다는 점이 오히려 학습이 불가능하다는 단점으로 바뀌어 잘 사용되지 않는다고 한다.<br>
> *(End-to-End 모델이 될 수 없다.)*
>
*([참고할 만한 블로그1](https://blog.naver.com/laonple/220925179894))*
*([참고할 만한 블로그2](https://blog.naver.com/laonple/220930954658))*


### 4) One-Stage & Two-Stage

![alt text](/assets/img/post/deeplearning_basic/onestage_twostage.png)

> | Detector | One-Stage | Two-Stage |
> | --- | --- | --- |
> | 탐지과정 | 1. 이미지를 grid로 나눔<br>2. 각 grid에서 물체위치 추정<br>3. 동시에 각 grid별 Classification | 1. Region Proposal로 물체위치 추정<br>2. 이 위치 안의 물체를 Classification |
> | 특징 | 실시간 탐지 가능 | 정확한 탐지 가능|
> | 종류 | YOLO | RCNN Family<br>SPP Net|
> 

---

## 2. Two-Stage Detection Model

### 1) R-CNN
<img src= "https://velog.velcdn.com/images/abrahamkim98/post/6f1f6c03-fe9f-4b94-bd23-da4cb05d072f/image.png" width=600>

> #### 동작과정
>
> ![alt text](/assets/img/post/deeplearning_basic/rcnn.png)
>
> 1. ROI추출(ex.Selective Search)<br>
>   : 먼저 이미지에 Selective Search같은 방법을 통해 약 2000개의 **ROI(Region of Interest)를 추출**한다.
>
> 2. Warping<br>
>   : 각 ROI는 모두 다른 크기를 갖고 있기 때문에 CNN Architecture에 넣기 위해 모두 동일한 크기를 갖도록 **Warping**을 해준다.
>
> 3. Feature추출<br>
>   : Warping된 ROI를 **CNN모델에 넣어 Feature를 추출**한다.
>
> 4. **Classification 및 Box Regression**<br>
>   ⅰ. **Classifier**: Feature를 SVM에 넣어 Classification<br>
>   *(SVM을 사용하는 이유는 그냥 과거에 나왔던 모델이라 그런듯 하다.)*<br>
>   ⅱ. **Box Regressor**: Bounding Box의 정확한 위치를 학습<br>
>   *(Selective Search는 물체의 대략적인 위치만 제공)*
>
> ---
> #### Box Regression
>
> | | 과정 |
> | --- | --- |
> | ![alt text](/assets/img/post/deeplearning_basic/box_regression.png)| 1. 파란색$(p_x, p_y, p_w, p_h) \rightarrow$ Region Proposal <br><br>2. 주황색$(b_x, b_y, b_w, b_h) \rightarrow$ Ground Truth<br><br> 3. $(t_x, t_y, t_w, t_h) \rightarrow$ Prediction<br> - $b_x = p_x+p_wt_x$<br>- $b_y = p_y+p_ht_y$<br>- $b_w = p_we^{t_w}$<br>- $b_h = p_he^{t_h}$ |
> 
#### 2. 문제점
- 모든 ROI에 대해 CNN모델을 통과시켜야 하기 때문에 해야하는 연산이 너무 많다.
>
>
- 다양한 크기의 ROI를 정해진 크기로 강제 Warping해주었기 때문에 성 하락 가능성이 존재한다.
>
>
>
- End-to-End학습이 불가능하다.
*(SVM사용, ROI와 CNN을 따로 학습시킴)*
>

### 2) SPPNet
<img src= "https://velog.velcdn.com/images/abrahamkim98/post/383d97a0-6bad-4191-831c-13b970e197ca/image.png" width=600>

>
SPPNet은 모든 ROI에 대해 CNN모델을 통과시켜야 하는 R-CNN의 단점을 먼저 CNN을 통해 Feature Map을 얻고 ROI를 결정하는 방식으로 해결하였다.
>
또한, 다양한 크기의 ROI를 하나의 정해진 크기로 강제 Warping해주어야 하는  R-CNN의 단점은 Spatial Pyramid Pooling을 수행하는 특별한 Layer를 통해 해결해 주었다.
>
---
#### 1. 동작과정
<img src= "https://velog.velcdn.com/images/abrahamkim98/post/d0ca0871-a951-489a-985f-3efe4dcb5ab7/image.png" width=400>
>
- 먼저 이미지를 CNN모델에 넣어 **Feature Map**을 얻는다.
>
>
- 이 Feature Map에 대해 Region Proposal 방법(Selective Search)을 적용해 ROI를 선별한다.
>
>
- 이 ROI들에 각각 **SPP(Spatial Pyramid Pooling) Layer**를 통해 고정된 크기의 Feature를 얻는다.
>
>
- 이렇게 얻은 Feature를 **SVM**에 넣어 Classification을 진행한다.
>
>
- Clasification이 완료된 물체를 골라 Bounding Box Regression*(Bounding Box의 정확한 위치에 대한 학습)*을 진행한다.
>
---
#### 2. Spatial Pyramid Pooling
<img src= "https://velog.velcdn.com/images/abrahamkim98/post/e6db3014-3bdd-4dd5-804d-1e272dbd1d16/image.png" width=350>
>
- Spatial Bins의 총 개수를 정한다. 이 개수는 입력으로 들어온 ROI를 표현하는 고정된 길이의 Feature가 된다.
*(ex. `21 bins` = `[4x4, 2x2, 1x1]`)*
>
>
- 앞서 정한 Spatial Bin을 사용해 Feature Map을 얻기 위해 적절한 Stride와 Window Size를 설정한다.
(ex. `ROI = 13x13`일 경우 `Stride = 4`, `Window Size = 5`로 설정하여 Max Pooing을 수행할 경우 `Spatial Bin = 3x3`을 얻을 수 있다.)
>
>
- 각각의 bin(`4x4`, `2x2`, `1x1`)에 대해 위의 과정을 반복하여 처음에 우리가 정했던 Spatial Bin을 모두 얻는다.
>
>
- 모든 Spatial Bin을 Flatten하고 연결시켜 고정된 길이의 Feature를 얻는다.
>
---
#### 3. 단점
>
- End-to-End학습이 불가능하다.
*(SVM사용, ROI와 CNN을 따로 학습시킴)*
>
---
([참고한 블로그](https://89douner.tistory.com/89))



### 3) Fast R-CNN

<img src= "https://velog.velcdn.com/images/abrahamkim98/post/7c3cc43e-11f8-481a-8bf8-4436785dfbb5/image.png" width=600>

>
SPPNet과 크게 다르지 않지만, ROI Projection이 존재하고 Spatial Pyramid Pooling대신 ROI Pooling을 사용한다는 점이 다르다.
>
---
#### 1. 동작과정
<img src= "https://velog.velcdn.com/images/abrahamkim98/post/ee521e2c-e705-4eac-86a8-ae1f9a2698b0/image.png" width=400>
>
- 먼저 이미지를 CNN모델에 넣어 Feature Map을 얻는다.
>
>
- 또 이미지에서 Selective Search를 통해 ROI를 구한다.
>
>
- 위에서 구한 Feature Map에 **ROI Projection**을 통해 ROI를 다시 얻는다.
>
>
- ROI Projection을 통해 얻은 ROI에 대해 **ROI Pooling**을 통해 일정한 크기의 Feature를 얻는다.
>
>
- ROI Pooling을 통해 구한 Output Feature로 Fully Connected Layer 적용 후 
*Softmax Classifier* 와 
*Bounding Box Regression (Bounding Box의 정확한 위치에 대한 학습)*
을 진행한다.
>
---
#### 2. ROI Projection
<img src= "https://velog.velcdn.com/images/abrahamkim98/post/cd20551c-3dbe-4511-94bb-7906ef776bc9/image.png" width=350>
>
Fast R-CNN에서는 Feature Map에서 Selective Search를 수행하는 SPPNet과는 달리, 이미지에서 Selective Search를 진행한 후 얻은 ROI의 위치를 사용한다.
>
즉, 이미지에서 얻은 ROI를 그대로 Convolution Feature Map에 Projection하여 진행한다.
>
---
#### 3. ROI Pooling
<img src= "https://velog.velcdn.com/images/abrahamkim98/post/95f6c020-5ea5-49c7-95f0-387f15c37431/image.png" width=600>
>
ROI Pooling은 SPP Layer에서
- 1개의 Pyramid Level만을 사용하고,
- Target bins = `7*7`
>
로 설정한 형태와 동일하다고 생각하면 된다.
>
*(참고: [ROI Pooling gif](https://upload.wikimedia.org/wikipedia/commons/d/dc/RoI_pooling_animated.gif))*
>
---
#### 3. 단점
>
- End-to-End학습이 불가능하다.
*(Selective Search를 사용)*
>
---
#### 4. 추가학습
- Hierarchical Sampling


### 4) Faster R-CNN

<img src= "https://velog.velcdn.com/images/abrahamkim98/post/73d0ff01-4853-44ee-80b5-57f842189c0e/image.png" width=600>

>
Faster R-CNN은 Fast R-CNN에서 Selective Search를 제외하고 Region Proposal Network라는 것을 추가하여 End-to-End 모델이 될 수 있도록 해 주었다.
>
---
#### 1. 동작과정
<img src= "https://velog.velcdn.com/images/abrahamkim98/post/d03084c2-32df-484f-a22a-ad069f070d4c/image.png" width=400>
>
- 먼저 이미지를 CNN모델에 넣어 Feature Map을 얻는다.
>
>
- Feature Map에서 **Region Proposal Network**와 **Non-Maximum Suppression** 통해 ROI를 얻는다. 
>
>
- 이렇게 얻은 ROI를 활용해 Fast R-CNN과 같이 동작하도록 구성한다.
(ROI Projection -> ROI Pooling -> Softmax Classification + Bounding Box Regression)
>
*(두개의 딥러닝 Network를 학습시켜야 하므로 Multi Task Loss를 사용한다..?)*
>
---
#### 2. Region Proposal Network(RPN)
>
<img src= "https://velog.velcdn.com/images/abrahamkim98/post/c56cd386-1c2b-484b-84f5-ddc0b35ab788/image.png" width=400>
>
- 우선 CNN모델에서 얻은 Feature Map에 대해 각 Cell을 Anchor로 지정한다.
그리고 이렇게 지정한 모든 Anchor를 중심에 배치하여 K개의 Anchor Box를 생성한다.
(K = len(Scale) \* len(ratio))
*(`Scale은 Box의 크기의 종류 수`, `ratio는 Box의 가로-세로 비율의 개수`)*
>
>
- 이때, 이 Anchor Box를 모두 사용하게 될 경우 너무 많은 ROI후보군이 생성된다.
*(예를들어, `64`\* `64` Feature Map에서 9개의 Anchor Box를 사용할 경우 3만6천개의 ROI가 발생한다.)*
>
>
- 즉, 배경에 대한 Anchor Box는 제외하기 위해 Anchor는 다음 두 역할에 대해 모두 학습이 되어야 한다.
 : 내용이 배경인지/물체인지 확인
 : Anchor Box와 실제 이미지의 Ground Truth의 위치가 같은지 확인
>
>
- 때문에 각 Cell별로 두개의 Layer의 입력으로 들어가게 된다. 
(단, `Anchor수`=`k`개 일때)
: 2k개의 Cls Layer =>`(배경o, 배경x)`
: 4k개의 reg Layer =>`(dx, dy, dw, dh)`
>
*(참고: k+Sigmoid로도 가능하지만 논문에서는 2k+Softmax 로 구현하였다고 함)*
>
<img src= "https://velog.velcdn.com/images/abrahamkim98/post/c9208e3b-9cf6-478d-90df-e3b37400448e/image.png" width=650>
>
*(참고그림)*
>
---
#### 3. Non-Maximum Suppression(NMS)
>
![](https://velog.velcdn.com/images/abrahamkim98/post/5dcc2dfd-edce-4085-8786-0ff33dbb4c93/image.png)
>
RPN의 결과로써 나온 ROI중에는 유사한 객체를 표현하는 ROI들이 여럿 존재하게 된다.
>
이를 막기 위해 Class Score를 기준으로 정렬한 후에, 중복된 영역이 많은 순서대로 ROI후보군을 삭제해 가면서 적절한 것을 Proposal 영역을 최소한으로 골라주는 알고리즘을 말한다.
>
([자세한 내용](https://velog.io/@abrahamkim98/Deep-Learning%EA%B8%B0%EB%B3%B8-4.-Object-Detectionensemble#1-nms))


---
다음장에 계속..

---
