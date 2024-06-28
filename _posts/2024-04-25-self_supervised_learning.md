---
title: "10. Self-Supervised Learning"
date: 2024-06-10 22:00:00 +0900
categories: ["Artificial Intelligence", "Deep Learning(Basic)"]
tags: ["sl", "self-supervised learning"]
use_math: true
---

## Self-Supervised with Transformer

---
### 1) SiT

(Self-Supervised Vision Transformer)

![alt text](/assets/img/post/computer_vision/sit.png)

Reconstruction(이미지 복원을 통한 학습)

### 2) BEIT

(Bert Pre-Training of Image Transformer)

![alt text](/assets/img/post/computer_vision/beit.png)

Mask Prediction(마스킹된 토큰을 예측함으로써 학습)

### 3) MAE

(Masked Autoencoders)

![alt text](/assets/img/post/computer_vision/masked_autoencoder.png)

Reconstruction(이미지 복원을 통한 학습)

---
## Multi-modal Self-Supervised Learning

### 1) VirTex

이미지와 Caption을 같이 학습하는 모델<br>
Caption이 가지는 Sematic Density덕분에 Contrastive Learning이나 Classification Pretraining에 비해 더 효과적임을 증명

![alt text](/assets/img/post/computer_vision/virtex.png)

1. 이미지 입력 & Feature 변환
2. Transformer를 사용해 Caption생성<br>
_(이 때, 양방향의 Caption을 생성하도록 설계함)_

3. 실제로 Caption이 더 많이질수록 더 좋은 성능을 가지는 것도 확인함

### 2) CLIP

![alt text](/assets/img/post/computer_vision/clip.png)

Contrastive Learning과 같이 학습한다.<br>
이미지와 Text에대해 Matching되는 부분은 Positive, 그렇지 않은 부분에 대해서는 Negative Sample로써 동작한다.

-> 매우 많은데이터가 필요하다는 단점

![alt text](/assets/img/post/computer_vision/clip_zeroshot.png)

pre-defined word embedding이 필요함