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


## 1. Stereo Matching

2개의 카메라를 사용하여 3D Depth Map을 추측하는 알고리즘