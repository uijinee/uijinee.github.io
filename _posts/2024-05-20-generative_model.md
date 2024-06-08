---
title: "8. Generative Model"
date: 2024-05-20 22:00:00 +0900
categories: ["Artificial Intelligence", "Deep Learning(Basic)"]
tags: ["vae", "gan"]
use_math: true
---

### Unsupervised Learning

생성형 모델은 Unsupervised Learning이라고 불린다.<br>
즉, Training Data에 Label이 존재하지 않다.

그렇다면 어떻게 학습을 진행할 수 있을까?<br>

생성형 모델의 가장 기본적인 Concept는 입력값에 대한 Latent Vector를 학습하는 것이다.<br>
즉, 입력값을 잘 표현할 수 있는 잠재적인(Latent) Vector를 학습하고 이를 생성에 활용할 수 있도록 한다.


$$
\mathbb{P(\mathbf{x})} = \int \mathbb{P}_w(\mathbf{x}, z) dz
$$

&#8251; $\mathbf{x}$: Input _(ex. 말, 단어, 목소리, ... 등 "형태")_<br>
$\quad z$: Latent Vector _(ex. 생각, ... 등 "개념")_

- Generation<br>
  : Latent Vector로 Input을 복원하는 것<br>
  $\rightarrow \mathbb{P}_w(\mathbf{x} \vert z) = \frac{\mathbb{P}_w(\mathbf{x}, z)}{\mathbb{P}(z)}$
 - Inference<br>
  : Input에서 Latent Vector를 추론하는 것<br>
  $\rightarrow \mathbb{P}(z \vert \mathbf{x}) = \frac{\mathbb{P}_w(\mathbf{x}, z)}{\mathbb{P(\mathbf{x})}}$<br>
  $\rightarrow $


---
## 2. VAE

## 3. GAN