---
title: "6. Segmentation(Instance, Panoptic)"
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

---
## 2. Instance Segmentation


---
## 3. Panoptic Segmentation