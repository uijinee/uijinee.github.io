---
title: "3. Self Attention"
date: 2024-03-19 22:00:00 +0900
categories: ["Artificial Intelligence", "Deep Learning(Basic)"]
tags: ["vision", "nlp", "rnn", "self-attention", "transformer"]
use_math: true
---

# Self Attention

## 1. BackGround

### 1) Concept

![alt text](/assets/img/post/deeplearning_basic/sequentialdata.png)

> RNN모델들은 Sequential Data를 Fixed Data로 바꾸는 모델이었다.
>
> 이번 장에서는 Sequential Data를 다른 Sequential Data로 변형해 주는 방법을 알아보자
>
> 이 때, 가장 큰 문제는 <u>입력의 길이와 출력의 길이가 다르기 때문에</u> 일반적인 RNN 모델들을 바로 사용할 수 없다는 것이다.
>
> 예를 들어 RNN을 사용하면 "넌 누구니?" 와 같이 2개의 입력이 들어갈 경우 "Who are you?" 와 같은 길이가 3인 출력이 나올 수 없다.
>
> ---
> **Encoder & Decoder**
>
> ![alt text](/assets/img/post/deeplearning_basic/encoder_decoder.png)
>
> 입출력 문장의 길이가 다를때 활용할 수 있는 대표적인 방법은 Encoder와 Decoder를 사용하는 것이다.
>
> Encoder는 입력할 Sequential Data를 하나의 정보로 압축하고,<br>
> Decoder를 통해 이 정보를 사용해 다시 Sequential Data를 만들어내게 된다.

## 2. Basic Model

### 1) Seq2Seq

![alt text](/assets/img/post/deeplearning_basic/seq2seq.png)

> **Fixed Size Context Vector**
> 
> Seq2Seq에서는 RNN기반의 Encoder를 통해 하나의 **고정된 크기의** Context Vector를 생성한다.
>
> Context Vector는 Start Token과 함께 Decoder에 들어가 단어를 하나 생성한다.<br>
> 이후부터는 생성된 단어를 다시 Decoder에 넣는 과정을 End Token이 나올때 까지 반복한다.
>
> - **단점**<br>
>  : 고정된 Size의 Context Vector를 사용하기 때문에 입력 문장의 길이가 길어질 경우, Vector가 이 정보를 표현할 수 없어 성능이 매우 떨어진다.
> 
> ---
> **Encoder Decoder**
>
> ![alt text](/assets/img/post/deeplearning_basic/seq2seq_encoderdecoder.png)
>
> Encoder와 Decoder로 Vanila RNN을 사용할 경우<br>
> Exploding/Vanishing Gradient에 빠질 가능성이 크다. 
>
> 따라서 보통 LSTM을 가지고 Encoder와 Decoder를 만든다.
>
> ---
> **Trick**
>
> 간단한 트릭이긴 하지만, 입력 데이터의 첫 글자는 뒤로 갈수록 변형되기 때문에,<br>
> Decoder에 처음으로 입력될 때, 이 첫 글자의 정보가 제대로 입력되지 않을 가능성이 있다.
>
> 위의 정보가 정확한 근거는 아니겠지만 실제로 입력 데이터를 반전시킬 경우 성능 향상이 있다고 한다.

### 2) Seq2Seq + Attention

![alt text](/assets/img/post/deeplearning_basic/seq2seq_attention.png)

> #### Purpose
>
> 1. Seq2Seq은 고정된 크기의 ContextVector를 사용해 모든 정보를 압축하여 정보의 손실(Bottleneck)이 발생한다.<br>
>  $\rightarrow$ **Non-Fixed size Context Vector**
>
> 2. RNN의 고전적인 문제인 Vanishing Gradient문제가 여전히 존재한다.<br>
>  $\rightarrow$ **Attention**<br>
>  　_(이 외에도 각각의 State에 가중치를 사용해 State별로 중요한 단어들을 학습할 수 있다는 장점등이 있다.)_
>
> ---
> #### 동작과정
>
> | ![alt text](/assets/img/post/deeplearning_basic/seq2seq_attention_process(1).png) | 1. **Initiate Decoder State**<br>　먼저 RNN을 반복하여 통과시켜<br>　Initial Decoder State를 만든다. |
> | ![alt text](/assets/img/post/deeplearning_basic/seq2seq_attention_process(2).png)<br>![alt text](/assets/img/post/deeplearning_basic/how_to_attention.png) | 2. **Attention Score**<br>　이 Decoder State(1개)와<br>　각 단계의 Encoder State(n개)로<br>　각각 Attention연산을 수행한다.<br><br>　　Attention시 사용할 수 있는 방법은 다음과 같다.<br>　　- Dot-Product Attention<br>　　- Bilinear Attention<br>　　- Multi-Layer Perceptron Attetion |
> | ![alt text](/assets/img/post/deeplearning_basic/seq2seq_attention_process(3).png) | 3. **Softmax**<br>　Attention 결과에 Softmax를 씌워 확률로 만든다.<br>　이를 통해, Decoder에서는 현재 어떤 단어에<br>　얼마나 집중해야 할 지 알 수 있다. |
> | ![alt text](/assets/img/post/deeplearning_basic/seq2seq_attention_process(4).png) | 4. **Attention Value**_(Context Vector)_<br>　입력 단어들을 통해 생성됐던 Hidden State에 대해<br>　Softmax를 통과한 Attention Score와<br>　가중합(Weighted Sum)한다. |
> | ![alt text](/assets/img/post/deeplearning_basic/seq2seq_attention_process(5).png) | 5. **Decoder**<br>　ⅰ) Attention Value의 결과를 반영하기 위해<br>　　기존의 Decoder State와 Concatenate한다.<br>　ⅱ) Concatenate된 신경망은 다시 신경망을 거쳐<br>　　기존의 Decoder State와 크기를 맞춰준다.<br>　ⅲ) 마지막으로 이 State와 Token을 사용해 <br>　　다음에 나올 단어를 예측한다. |
> 
> ---
> #### Teacher Forcing
>
> Seq2Seq에서는 Decoder의 결과(출력된 값)을 다시 다음 Decoder의 Token으로 사용한다.<br>
> 이때, 학습이 되기 전에는 Prediction을 수행하면 우리가 원하지 않을 결가 나온다는 것이 자명하다.<br>
>
>> 즉, 다음 Token부터는 엉뚱한 값을 넣게 된다.
> 
> 이를 방지하기 위해 학습시에는 예측 값을 Token으로 사용하는 것이 아닌, 정답값을 따로 넣어 Token으로 사용한다.
> 
> ---
> #### 단점
>
> Attention을 활용해 많은 장점을 갖게 되었지만 결국 RNN을 활용해 학습 시켜야 한다.
>
> 즉, 입력을 Sequential하게 입력해 학습시켜야 하기 때문에 학습 시간이 오래 걸리고, 입력의 길이가 길어지는데에도 한계가 존재한다.
> 
> ---
> [참조한 영상](https://www.youtube.com/watch?v=WsQLdu2JMgI&t=404s)
> 
> [참고한 블로그](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#)

### 3) Show, Attend, and Tell

![alt text](/assets/img/post/deeplearning_basic/show_attend_tell.png)

> #### Purpose
>
> 1. 이전 논문인 Show and Tell에서는 기존의 seq2seq과 비슷한 문제들을 갖고 있었다.<br>
> (항상 같은 Context Vector를 참고하여 캡셔닝하는문제)<br>
>  $\rightarrow$ **Attention**
>
> ---
> #### 동작과정
> 
> | ![alt text](/assets/img/post/deeplearning_basic/show_attend_tell_procedure(1).png) | 1. **Initial Decoder State**<br>　ⅰ) CNN으로 Feature Map을 뽑아낸다.<br>　ⅱ) Feature Map을 Grid로 나눈다.<br>　ⅲ) Global Average Pooling을 통해 Decoder State를 만든다. |
> | ![alt text](/assets/img/post/deeplearning_basic/show_attend_tell_procedure(2).png) | 2. **Alignment Score**(_($\approx$ Attention Score)_<br> 　Seq2Seq와 마찬가지로 Attention을 통해<br>　$s_0$와 $h_{i, j}$간의 유사도 점수를 구해준다.|
> | ![alt text](/assets/img/post/deeplearning_basic/show_attend_tell_procedure(3).png) | 3. **Context Vector**<br>  　Seq2Seq와 마찬가지로 Softmax를 통해<br>　Alignment Weights를 구하고 기존의 Feature Map($h_{i, j}$)와<br>　Weighted Sum을 통해 Context Vector를 구한다. |
> | ![alt text](/assets/img/post/deeplearning_basic/show_attend_tell_procedure(4).png)| 4. **Predict**<br>　ⅰ) 기존의 Decoder State $s_0$와 Context Vector를 합친다.<br>　ⅱ) Start Token인 $y_0$를 사용해 Decoder에서<br>　　Prediction을 수행한다. |
> | ![alt text](/assets/img/post/deeplearning_basic/show_attend_tell.png) | 5. **반복**<br>　위의 과정을 반복 수행하며 문장을 만들어 나간다.|
>
> ---
> #### Alignment Weight 시각화
> 
> ![alt text](/assets/img/post/deeplearning_basic/alignment_weight.png)

---
## 3. Attention Layer

위의 Basic Model과 같이 Attention Model이 주목받기 시작하자, 이 후부터는 이 Attention과정을 CNN과 같은 하나의 Layer로 취급하여 모델을 설계하기 시작한다.

| | Attention Layer | Show, Attend and tell |
|:---:| --- | --- |
| | ![alt text](/assets/img/post/deeplearning_basic/attention_layer.png) | ![alt text](/assets/img/post/deeplearning_basic/show_attend_tell_procedure(3).png) |
| Query<br>_(알고싶은 대상)_ | $Query$ | $s_0$ |
| Key<br>_(비교할 대상)_| $Key$<br>_(InputVector $X$에 MLP를 적용해 생성)_ | {$h_{i, j}$} |
| Context Vector<br>생성 도구 | $Value$<br>_(InputVector $X$에 MLP를 적용해 생성)_  | {$h_{i, j}$} |
| Output<br>_(Context Vector)_ | $Y=AV$<br>_(Weighted Sum)_ | $C=AH$<br>_(Weighted Sum)_ |
| Similarity함수 | Scaled Dot-Product<br>$E=\frac{QK^T}{\sqrt{D_Q}}$<br>_(Q, K의 길이가 클수록 Softmax에 의해<br> 0에 가까운 Gradient가 많이 발생한다.)<br> (Gradient Vanishing현상)_ | - Dot-Product Attention($E=QK^T$)<br> - Bilinear Attention<br> - Multi-Layer Perceptron Attetion |

_(Attention Layer는 총 2개의 Learnable Parameter를 갖는다.)_

### 1) Transformer

> 2017년도에 Attention is All You Need라는 제목으로 발표된 논문에서는 RNN Cell을 아예 제거하고,<br>
> 오직 위의 Attention Layer만을 사용해 학습시키는 Transformer구조가 제시된다
> 
> _(Attention is All you Need라니... YOLO와 더불어 자극적인 논문 제목 중 하나인 것 같다.)_
>
> ---
> #### Purpose(Self Attention Layer)
>
> 1. Attention Layer의 체계적인 Design<br>
>  $\rightarrow$ **Self Attention Layer**
>
> | Layer |  | 특징 |
> | --- | --- | --- |
> | **Self Attention** | ![alt text](/assets/img/post/deeplearning_basic/self_attention_layer.png) | 1. Query를 별도로 입력해주는 것이 아닌<br>　InputVector$X$로부터 MLP를 적용해 생성한다.<br> 　$\Rightarrow$ 3개의 Learnable Layer<br><br>2. 벡터들의 집합으로서 동작한다.<br>　$if (X_1, X_2, X_3 \rightarrow X_3, X_1, X_2)$<br>　$\Rightarrow (Y_1, Y_2, Y_3 \rightarrow Y_3, Y_1, Y_2)$<br>　$\therefore$ InputVector의 순서정보를 알지 못한다.<br>　$\Rightarrow$ Positional Encoding이 필요하다.<br>　　![alt text](/assets/img/post/deeplearning_basic/positional_embeding.png) (concat)  |
> | **Masked**<br> **Self Attention** | ![alt text](/assets/img/post/deeplearning_basic/masked_self_attention.png) | 기존의 Encoder Decoder 구조의 모델들은<br> 입력값을 순차적으로 전달받아 $t+1$시점의<br> 예측을 위해 $t$까지의 데이터만 쓸 수 있었다.<br><br> 하지만 Transformer는 한번에 모든 입력을<br>받기 때문에 과거 시점의 입력을 예측할 때<br> 미래시점의 입력도 참고할 수 있다. <br><br> 이를 방지하기 위해 사용하는 것이<br> **<u>Look a Head Mask</u>**이다. |
> | **Multi-Head**<br> **Self Attention** | ![alt text](/assets/img/post/deeplearning_basic/multi-head_self_attention.png) | n개의 **Self Attention Layer**를<br> **<u>Parallel</u>**하게 동작하도록 구성한 Layer이다. |
>
> ---
> #### Transformer
> 
> | | Transformer Block | Transformer Architecture |
> | --- | --- | --- |
> | 그림 | ![alt text](/assets/img/post/deeplearning_basic/transformer_block.png) | ![alt text](/assets/img/post/deeplearning_basic/transformer_architecture.png)  |
> | 특징 | - **Layer Normalization**<br> | - Encoder Decoder Design<br>- Sequence of Transformer Block|
> 

### 2) VIT(Vision Transformer)

![alt text](/assets/img/post/deeplearning_basic/vit.png)

> #### Purpose
> 
> 1. Attention Module을 Image에도 적용시켜보고자 함<br>
>  ⅰ) 방법1: CNN Architecture사이에 Self-Attention Layer를 넣는다<br>　$\rightarrow$ 그래도 CNN인건 변하지 않는다.<br>
>  ⅱ) 방법2: Pixel의 관계를 계산할 때 Convolution대신 Self-Attention을 넣는다<br>　$\rightarrow$ 구현이 힘들고 성능도 별로<br>
>  ⅲ) 방법3: image를 Resize $\rightarrow$ flatten 하고 이 Image에 대해 Self-Attention을 한다<br>　$\rightarrow$ resize결과가 $R \times R$일경우 Self Attention에서 $R^2 \times R^2$ 의 메모리가 필요하다.<br>
> **ⅳ) 방법4:** Image를 Patch별로 나눈 후 linear projection하고 이것들에 대해 Self-Attention을 한다<br> 　$\rightarrow$ **Vision Transformer**
> 
> ---
> #### 동작과정
>
> | ![alt text](/assets/img/post/deeplearning_basic/vit_process(1).png) | 1. **Linear Projection**<br>　MLP나 CNN을 사용해 각 Patch들을 $D$차원 Vector로 Flatten한다.<br>　_(각 패치의 크기가 $16 \times 16 \times 3$일 때,<br>　mlp weight의 경우 $16 \times 16 \times 3 \times D$의 크기를 갖는다)_ |
> | ![alt text](/assets/img/post/deeplearning_basic/vit_process(2).png) | 2. **Positional Embedding**<br>　각 Patch에 Positional Embedding을 적용한다.<br>　이 Embedding Vector들은 독립적으로 학습된다. |
> | ![alt text](/assets/img/post/deeplearning_basic/vit_process(3).png) | 3. **Transformer**<br>　Transformer를 통해 Patch들을 가공한다.<br>　- 이때, 당시 nlp에서 Convention이던 Classification Token을 추가<br>　- 대응하는 Output은 전체 Feature를 표현하는 Global Vector |
>

### 3) Swin Transformer

> #### Purpose 
>
> 1. ViT는 CNN에 비해 Inductive Bias가 적다<br>　$\rightarrow$즉, 학습을 위해서 매우 많은 데이터가 필요하다.
>
> 2. CNN은 깊어질수록 Resolution은 감소하고 Channel은 증가하는 Hierarchical구조를 갖는다.<br>　$\rightarrow$ 반면에 ViT는 모든 Block이 같은 Resolution을 갖는 Isotropic Architecture를 갖는다.
> 

### 4) MLP Mixer


<!--
### 1) Idea

<img src="https://velog.velcdn.com/images/abrahamkim98/post/ded2c55a-12f5-4232-926e-a8d0cbb0f721/image.png" width=450>

>
Transformer는 RNN모델들을 아예 제거하고 Attention 기법으로만 Input Data를 해석한다.
>
이 때, RNN이 사라지면서 문장에서 단어의 순서를 고려할 수 없게 되었기 때문에 Input값에 위치 정보도 추가해주도록 하는 Positional Encoding이라는 방법을 사용한다.
>
---
#### Transformer의 장점
1. 계산의 효율성
: 기존의 RNN기반의 복잡하고 시간이 걸리는 모델에서 벗어났다.
>
>
2. Saturating Performance가 없음
: 데이터셋의 크기가 많더라도 성능 향상의 한계가 있다는 것을 Saturating Performance라고 한다.
>
 Transformer는 데이터 셋의 크기가 적으면 그 성능이 매우 떨어지지만 그만큼 많으면 많을수록 성능이 다른 모델들보다 좋아진다.
>
>
3. 큰 싸이즈의 모델도 훈련이 가능하다.
: 데이터를 순서대로 처리하는 RNN과는 달리 한번에 처리하기 때문에 모델의 크기와 상관없이 사용 가능하다.
*(단, 그만큼 많은 용량의 메모리가 필요하다.)*
>
위와 같이 많은 장점들을 가지고 있기 때문에 Transformer는 NLP뿐만 아니라 CV에서도 사용하고자 한다. 대표적으로 Vision Transformer(VIT)가 있다.
>
---
[참고할 만한 블로그](https://ahnjg.tistory.com/57)


### 2) Positioinal Encoding
<img src="https://velog.velcdn.com/images/abrahamkim98/post/cea83a90-7c5d-42eb-8da9-f7f8433c945f/image.png">

>
앞서 말했듯이 Transformer는 RNN을 사용하지 않기 때문에 Positional Encoding이 필요하다.
>
이때, 해당 논문에서는 아래의 그림과 같은 Cosine과 Sine함수를 이용해 Positional Encoding을 한다.
>
<img src="https://velog.velcdn.com/images/abrahamkim98/post/da6dbfe8-fccc-48c4-b47d-5cca5f5ed2b9/image.png">
>
이때, Cosine과 Sine은 주기함수여서 위치값이 겹치지 않을까 생각할 수 있지만, 여러 주기의 Sine과 Cosine함수를 동시에 사용함으로써 해당 문제를 해결하였다.
>
예를 들어 위의 그림에서 서로 다른 두 위치를 정할 경우 해당 위치에서의 Sine과 Cosine함수의 집합은 다른 위치의 벡터와 전혀 다르다는 것을 확인할 수 있다.
>
---
자세한 내용은 다음의 [블로그](https://www.blossominkyung.com/deeplearning/transfomer-positional-encoding)를 참조

### 3) Encoder - Multi Headed Attention

<img src="https://velog.velcdn.com/images/abrahamkim98/post/02874976-0e85-4d40-82cc-5876c969c1ed/image.png" width=300>

>
---
### -- Self Attention --
>
---
#### Preview
<img src="https://velog.velcdn.com/images/abrahamkim98/post/9e1bbb92-bb53-4e2e-8ad5-9ea2d82f7838/image.png" width=300>
>
문장 속에서 어떤 단어를 이해할 때에는 그 단어 자체만으로 이해하면 안되고 그 단어가 문장에서 어떤 역할을 하는지, 즉 다른 단어들과 어떤 연관을 갖는지 알아야 한다.
>
예를들어, 위의 그림에서는 문장 속에서 `it`이 `The Animal`과 가장 큰 관련성을 가진다는 것을 나타내 준다.
>
자세한 과정을 설명하기에 앞서 그 과정을 요약하자면 다음과 같다.
- 각 단어에 대해 Query, Key, Value라는 정보를 만든다.
- 이것과 다른 단어의 Key, Value를 곱해 다른단어들과의 연관성을 구한다.
- 다른 단어들과의 연관성 정보를 합해 해당 단어의 Encoding Vector를 구한다.
>
[참조한 영상](https://www.youtube.com/watch?v=mxGCEWOxfe8)
>
---
>
1. **각 단어별로 Query, Key ,Value Vector를 만든다.**
   - Vector의 크기는 하이퍼 파라미터로, 사용자가 직접 설정한다.
   - Query와 Key는 항상 차원이 같아야 하지만 Value는 같을 필요는 없다.
>
<img src="https://velog.velcdn.com/images/abrahamkim98/post/72f28be1-d9e4-47d4-bfa5-2876ae3e05c8/image.png" width=300>
>
>>Query, Key, Value의 의미
`Query`: 단어에 대한 Embedding값?
`Key`:
`Value`:
>
---
2. **각 단어의 Query에 대해 나머지 단어의 Key를 곱해 `Attention Score`를 만든다.**
   - Normalize
   : Score Vector를 구한 후 `sqrt(Key Vector의 Dimension)`로 나누어 SoftMax연산시 값이 잘 나오도록 해준다.
>
<img src="https://velog.velcdn.com/images/abrahamkim98/post/e847fe9e-d12b-4b07-a196-e274066baf0b/image.png" width=300>
>
>> 이 Score는 후에 Attention Weight로 변해 각 단어가 다른 단어에 대해 얼마나 집중해야 하는지 결정하는 역할을 한다.
>
----
>
3. **Score Vector를 Normalize해주고 `Attention Weight`를 구한다.**
   - Attention Weight
   : Normalize한 Score에 Softmax연산을 취해준다.
>
<img src="https://velog.velcdn.com/images/abrahamkim98/post/cfbd7a0e-f819-4dda-a8d7-bf2e53bd6fea/image.png" width=350>
>
>> 이 Attention Weight는 현재 단어에 대해서 다른 각 단어들에 대한 가중치를 표현하게 된다.
>
----
>
4. **Attention Weight에 각각의 단어의 Value를 곱해준다.**
<img src="https://velog.velcdn.com/images/abrahamkim98/post/3bf09f33-fa46-4261-b7cc-6cdb7e5df86c/image.png" width=450>
>
>> 여기서 현재 단어가 각 단어의 정보를 얼만큼 표현해야 하는지 결정된다.
>
---
>
5. **위에서 구한 값들을 모두 더해 해당 단어에 대한 Encoding Vector를 구한다.**
<img src="https://velog.velcdn.com/images/abrahamkim98/post/e5936ed9-c24f-4703-bf9e-94ac91aab873/image.png" width=530>
>> 이제 우리는 각 단어에 대한 Encoding Vector를 얻었다.
>>
이 때, 이 Encoding Vector는 우리가 미리 설정한 Value Vector와 차원이 같다는 것을 유의해야 한다.
>
---
#### Self Attention 그림 요약
>
<img src="https://velog.velcdn.com/images/abrahamkim98/post/61a8cb89-99b5-44dd-a592-0daf39fde2ae/image.png" width=480>
>
---
### -- Multi Headed Attention --
---
<img src="https://velog.velcdn.com/images/abrahamkim98/post/49a75fd6-7d0c-447e-bb7d-0db1cc4c7b1c/image.png" width=480>
>
사람의 문장은 모호할 때가 훨씬 많고, 다양한 해석의 가능성이 존재한다.
>
따라서 본 논문에서는 위의 각 단어에 대해 Self Attention의 과정을 여러개의 Query-Key-Value쌍을 만들어 수행하도록 하여 여러 관점에서 그 문장을 해석하도록 해 주었다.
>
그림에서 보면 알 수 있듯이, 이 여러 Self Attention의 결과를 Concatenate한 후, 다시 Linear연산을 통해 다음 Encoder에 들어갈 수 있도록 크기를 조절해 주는 것을 확인할 수 있다.
>
---
*(Multi Headed Attention 과정은 병렬적으로 처리되기 때문에 계산 속도에 대해서는 거의 영향을 주지 않는다.)*

### 5) Encoder
<img src="https://velog.velcdn.com/images/abrahamkim98/post/ff997972-fab7-4eff-9ae3-8ba7ebf43911/image.png" width=350>

>
#### Feed Forward
>
<img src="https://velog.velcdn.com/images/abrahamkim98/post/f7549bf5-1021-4866-8d1d-a0d002366824/image.png" width=250>
>
Multi Headed Attention의 결과값은 다시 Fully Connected Layer를 거쳐 Encoder의 최종 출력으로 전달 되게 된다.
>
>> **이때, Encoder의 최종 출력의 모양은 최초 입력의 모양과 같다는 점을 기억하자.**
>>
*(각 단어의 Encoding Vector는 Value Vector의 모양과 같았다는 점과 헷갈리지 말자)*
>
---
>
#### Residual Connection
>
<img src="https://velog.velcdn.com/images/abrahamkim98/post/10ac81ae-8ae4-4446-9f9b-434dda921140/image.png" width=450>
>
word embeding + positional encodnig
>
해당 모델을 학습시키다 보면 역전파과정에 의해 이 Positioinal Encoding의 정보가 많이 손실 될 가능성이 크다.
>
이를 보완하기 위해 Residual Connection을 사용해 주었다
>
---
#### Encoder Layer
>
<img src="https://velog.velcdn.com/images/abrahamkim98/post/6521c3c1-3a96-4a87-bc9a-d953028439d0/image.png" width=550>
>
Ecoder의 입력과 출력의 모양이 같다는 점을 이용하여 위 그림과 같이 서로다른 Encoder여러개를 쌓아줄 수 있다.
>
본 논문에서는 6개의 층을 쌓았다고 한다.




### 6) Decoder - Masked Self Attention
>
Encoder의 경우 예측을 위해 사용할 수 있는


### 7) Decoder - Masked Multihead Attention
>

### 8) Decoder
>
#### Feed Forward
>



---
---
# 코드

-->