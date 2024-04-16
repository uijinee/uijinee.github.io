---
title: "5. Segmentation(Semantic)"
date: 2024-03-29 22:00:00 +0900
categories: ["Artificial Intelligence", "Deep Learning(Basic)"]
tags: ["cnn", "segmentation"]
use_math: true
---

# Segmentation

![alt text](/assets/img/post/deeplearning_basic/segmentation_history.png)

## 1. Background

### 1) Segmentation의 종류

### 2) Upsampling

> #### UnPooling
>
> | 1. Nearest Neighbor | 2. Bed Of Nails | 3. Max Unpooling |
> | --- | --- | --- |
> | ![alt text](/assets/img/post/deeplearning_basic/nearest_neighbor.png) | ![alt text](/assets/img/post/deeplearning_basic/bed_of_nails.png) | ![alt text](/assets/img/post/deeplearning_basic/max_unpooling.png) |
> | 자신의 값으로 채우는 방법 | 모두 0으로 채우는 방법 | max값의 index를 기억했다가<br> Unpooling시 그 자리에 넣는 방법<br>_(나머지는 0)_ |
>
> | 4. Bilinear Interpolation | 5. Bicubic Interpolation |
> | --- | --- |
> | ![alt text](/assets/img/post/deeplearning_basic/bilinear_interpolation.png) | ![alt text](/assets/img/post/deeplearning_basic/bicubic%20interpolation.png) |
> | 선형 보간법<br>　- 두 점 사이의 점에 대해 선형 함수를<br>　사용해 계산하는 방법(내분 점)<br><br>ⅰ. 미지수를 설정해 1차함수 설정<br>ⅱ. 미지수가 2개이므로 2개의 점에 대해 계산 | 삼차보간법<br>　- 두 점 사이의 점에 대해 3차함수를<br>　사용해 계산하는 방법<br><br>ⅰ. 미지수를 설정해 3차함수 설정<br>ⅱ. 미지수가 4개이므로 4개의 점에 대해 계산 |
> 
> | 6. Transpose Convolution |
> | --- |
> | ![alt text](/assets/img/post/deeplearning_basic/transposed_convolution.png) |
> | Learnable Upsampling 방식으로<br>적절한 Padding, Stride, Convolution을 이용해 Unsampling하는 방법 |
> | _(Overlap Issue)_<br> ![alt text](/assets/img/post/deeplearning_basic/overlap_issue.png)<br> 일반적인 Transposed Convolution은 일정 간격마다 밝기가 달라지는 Overlap Issue가 발생한다.<br> 이 단점을 해결하기 위해 주로 Bilinear등으로 Upsampling을 한 후에 Convolution을 수행한다. |
> 
> _(UnPooling은 Channel By Channel로 진행된다.)_


## 2. Semantic Segmentation


### 1) FCN(Fully Convolution Network)

![alt text](/assets/img/post/deeplearning_basic/fcn.png)

_FCN은 Fully Convolutional Networks의 약자로 FC Layer를 사용하지 않고 Convolution Layer만을 사용하여 이미지를 처리하는 Network라는 특징이 있다._

> #### Purpose
>
> ![alt text](/assets/img/post/deeplearning_basic/convolutionalization.png)
>
> 1. 만약 $3 \times 3$ Convolution만을 사용해 Spatial Dimension을 변화시키지 않도록<br>
>   Model을 설계하면 Layer대비 Receptive Field의 크기가 작아지기 때문에<br>
>   모델이 이미지의 전반적인 Context를 파악할 수 없다.<br>
>   $\rightarrow$ **Encoder-Decoder Design**
>
> 2. 일반적인 Classification 모델은 마지막에 Fully Convolution Layer를 사용한다.<br>
>   이때, Flatten이 필요하기 때문에 위치정보를 유지할 수 없다.<br>
>   $\rightarrow$ **Convolutionalization** <br>
>   　(이미지를 줄이지 않고 $1\times1$ Convolution만을 사용해 채널크기만 변경하는 것)
>
> ---
> #### 동작과정
> 
> | ![alt text](/assets/img/post/deeplearning_basic/fcn_procedure.png)
>
> | Encoder | Decoder |
> | --- | --- |
> | $3 \times 3$ Convolution을 사용해 Down Sampling한다.<br><br> &#8251; **Skip Connection**<br>마지막 Feature Map은 큰 Receptive Field를 가짐<br> $\rightarrow$ Detail한 정보를 갖지 못함 <br>즉 이를 위해 Skip Connection이 필요하다.<br>　ⅰ. **Concatenate**<br>　　: Elementwise 덧셈인 FPN과 다르게 합침<br>　ⅱ. **Upsampling**<br>　　: 크기를 맞추기 위해 Bilinear Upsampling | FCN에서는 Bilinear로 한번에 Up Sampling해 주었다.<br><br> _(FCN-32s는 Decoder에서 32배 Upsampling했다는 뜻)_ |
>
> ---
> #### 문제점
> 
> 객체의 크기가 크거나 작은 경우 여전히 예측을 잘 하지 못함<br>
> 객체의 디테일한 모습을 찾지 못함


<!--

---
## 2. DeconvNet

<img src="https://velog.velcdn.com/images/abrahamkim98/post/adc2842e-bf56-48b8-94e1-fe712b01cf42/image.png">

### 1) Encoder
<img src="https://velog.velcdn.com/images/abrahamkim98/post/29ef1269-e3e5-44ab-bdad-e6922c85aa79/image.png" width=500>

>
DeconvNet은 Encoder와 Decoder를 통해 위에서 설명한 FCN의 한계 극복하고자 한 Architecture이다.
>
FCN은 Encoder를 통해 Feature Map을 추출하고 이후에 바로 Transposed Convolution을 활용해 Upsampling을 진행해 객체의 위치를 파악했지만, DeconvNet에서는 Decoder라는 Layer를 활용해 Upsampling을 차근차근 진행하게 된다.
>
---
#### 1. BackBone
Encoder의 전체적인 구조는 앞의 FCN과 마찬가지로 VGG16을 사용하게 된다.
>
이때, VGG16은 Classification Task를 해결하기 위해 나왔던 Architecture인 만큼 마지막에 `1x1 Feature Map`을 추출하기 위해 `7x7 Convolution`과 `1x1 Convolution`을 활용하게 되는데 이를 그대로 활용했다는 점을 유의하자.
>
---
#### 2. SegNet
<img src="https://velog.velcdn.com/images/abrahamkim98/post/e894b466-8f7c-4927-ab83-6b82d008c871/image.png" width=500>
>
이 DeconvNet의 Encoder-Decoder Architecture를 거의 그대로 활용하고, Real Time Semantic Segmentation을 하기 위해 몇몇 Layer를 변경하도록 고안된 Architecture이다.
*(즉, 속도를 위해 정확도를 높여주는 몇몇 요소들을 제거한 것 같다.)*




### 2) Decoder
<img src="https://velog.velcdn.com/images/abrahamkim98/post/5bdbd134-3bc6-4e8a-8d55-8c548f2d9d78/image.png">

>
Decoder는 크게 2가지의 작업을 하게 되는데 각 잡업이 하는 일은 다음과 같다.
- UnPooling: 디테일한 경계를 복원
- Transposed Convolution: 전반적인 내용을 복원
>
---
#### 1. UnPooling
<img src="https://velog.velcdn.com/images/abrahamkim98/post/356408b7-39b2-48a8-a079-ad8a32837643/image.png" width=500>
>
UnPooling은 앞서 수행했던 MaxPooling이전의 값을 복원하는 역할을 한다고 생각하면 된다.
>
이때, 복원을 위해서는 위의 그림과 같이 Max값의 Indice가 필요하다.
>
즉, 해당 Indice에 해당하는 곳에 현재 알고있는 Max값을 채워놓고 나머지는 0으로 채워넣게 된다.
>>
**즉, UnPooling은 Object의 경계값을 알아내는 역할을 하게된다.**
>
---
#### 2. Transposed Convolution
>
<img src="https://velog.velcdn.com/images/abrahamkim98/post/480538f4-dc1c-444a-a312-c25ce99b7cef/image.png" width=500>
>
UnPooling의 결과는 어쨌든 0인 값을 가지고 있는 Sparse한 Activation Map을 가지게 된다.
>
이 부분을 채워주는 역할을 Transposed Convolution이 수행하게 된다.
>>
**즉, Transposed Convolution은 UnPooling의 경계값을 활용해 Object의 안의 내용을 복원하는 역할을 하게 된다.**
>
---
#### 3. Analysis Of Deconvolution Network
<img src="https://velog.velcdn.com/images/abrahamkim98/post/c19856ec-8816-43ab-b465-bc6e6928be60/image.png" width=500>
>
- UnPooling *(`(c)`, `(e)`, `(g)`, `(i)`)*
: 위의 해당 그림을 보면 알 수 있듯이 UnPooling작업은 경계값을 포착하고 있는 것을 확인할 수 있다. 
>
>
- Transposed Convolution *(`(b)`, `(d)`, `(f)`, `(h)`, `(j)`)*
: 마찬가지로 Transposed Convolution은 경계값 내부의 빈 공간을 복원하고 있는 것을 확인할 수 있다.
>
>
- Layer
: 위의 그림을 자세히 살펴보면 `(b)`와 같은 얕은 층의 경우에는 전반적인 모습을 잡아내고 있지만 , `(j)`와 같이 깊은 층의 경우에는 구체적인 모습까지 잡아내고 있는 것을 확인할 수 있다.



### 3) 코드 참조 설명
>
### Encoder
<img src="https://velog.velcdn.com/images/abrahamkim98/post/a122812d-dac0-4207-a89b-c237114a7923/image.png" width=400>
>
#### 1. Conv
>>
```py
self.Conv = nn.Sequential(
	nn.Conv2d(),
    nn.BatchNorm2d(),
    nn.ReLU()
)
```
- *이때, Conv블럭에서는 `3x3`의 Kernel Size와 `1`의 Padding을 주어 이미지 크기(Resolution)의 변화가 발생하지 않는다.*
>
#### 2. Pooling
>>
```py
self.Pooling = nn.MaxPool2d(kernel_size=2 ,stride=2, return_indices=True)
>>
# out, Pooling_indice = self.Pooling(Out)
```
- *`return_indices=True`설정을 통해 MaxPooling시의 위치 정보를 저장해두고 후에 Unpooling시에 활용하게 된다.*
>
#### 3. Encoder Block
>>
```py
Encoder = nn.Sequential(
	self.Conv(),
    self.Conv(),
    self.Pooling()
)
```
- *Vgg16과 같이 처음 2 Block은 Conv블록을 2개씩를 활용하고 마지막 3 Block은 Conv블록을 3개씩 활용한다.*
>
---
### Decoder
<img src="https://velog.velcdn.com/images/abrahamkim98/post/8e5b50c7-3714-4d75-a4c7-c061edf19e01/image.png" width=400>
>
#### 1. UnPooling
>>
```py
self.UnPooling = nn.MaxUnpool2d(kernel_size=2, stride=2)
>>
# out = self.UnPooling(out, Pooing_indice)
```
>>
- *Unpooling을 위해서 위의 Pooling작업시 `return_indice=True`로 설정해 indice값을 같이 넣어주어야 한다.*
>
#### 2. Deconv
>>
```py
self.Deconv = nn.Sequential(
	nn.ConvTransposed2d(),
    nn.BatchNorm2d(),
    nn.ReLU()
```
>>
- *마찬가지로 Deconv블럭으로 인해 Sparse한 Feature Map을 Dense하게 바뀌게 되지만 Resolution의 변화는 없다.*
>
#### 3. Decoder
>>
```py
Decoder = nn.Sequential(
	self.UnPooling(),
	self.Deconv(),
	self.Deconv()
)
```
- *마찬가지로 그림과 같이 Deconv블럭의 수를 적절히 조절하여 사용한다.*
>

---
## 3. FC DenseNet
<img src="https://velog.velcdn.com/images/abrahamkim98/post/3c6b62e7-eb44-46f3-9ab5-0ef6d9892e7d/image.png" width=400>


### 1) Dense Block
<img src="https://velog.velcdn.com/images/abrahamkim98/post/64649bff-c597-4282-8b3d-231e3fd99579/image.png" width=120>

>
#### 1. DenseNet
앞서 배웠던 [DenseNet](https://velog.io/@abrahamkim98/Deep-Learning-4.-CNN#6-densenet)에서는 Skip Connection을 독특하게 적용하는 Dense Block을 활용하여 Feature Map의 Propagation을 강화하였었다.
>
FC DenseNet에서는 이 Dense Block을 활용하여 low Level의 Feature Map이 Output에 잘 전달되어 객체의 디테일한 모습을 잘 포착하고자 하였다.
>
앞서 배웠던 내용이긴 하지만 Dense Block의 가장 큰 특징을 요약해보면 다음과 같다.
>
- Feed Forward시 각 Layer들을 다른 모든 Layer들과 연결한다.
- Skip Connection시에 Resnet에서 제안했던 Addition방법을 하는 대신에 Concatenation을 활용한다.


### 2) Architecture
<img src="https://velog.velcdn.com/images/abrahamkim98/post/e1c85740-6037-4037-8a9f-a55764bbc965/image.png" width=350>

>
#### 1. Encoder & Decoder
DeconvNet과 마찬가지로 Encoder와 Decoder를 활용하여 Output을 추출하게 된다.
>
이때, 각 Block은 Dense Block으로 구성되있고, 그림을 보면 알 수 있듯이 Encoder와 Decoder는 Skip Connection으로 다시한번 연결되고 있음을 확인할 수 있다.

-->
---
###  4) U-Net

<img src="https://velog.velcdn.com/images/abrahamkim98/post/8b310064-5c52-4787-9659-5e4b4a6e7984/image.png" width=600>

>
U-Net모델은 FCN의 Skip Connection구조를 좀더 잘 활용하도록 설계하여 성능을 향상시켰다.
>
---
#### Contracting Path
![](https://velog.velcdn.com/images/abrahamkim98/post/00ad8387-684a-4090-91f0-72cbc6f7a2e8/image.png)
- Convolution Layer
- Receptive Field의 크기를 키우기 위해 PoolingLayer를 사용해 해상도를 낮추는 대신 채널의 수를 늘림
- 최종적으로 작은 Activation Map을 얻음
>
---
#### Expanding Path
![](https://velog.velcdn.com/images/abrahamkim98/post/a7e5490b-a865-46ba-b133-30c717632ed4/image.png)
- 위에서 얻은 Activation Map을 UpSampling한다.
(Upsampling할때 적절한 Stride와 Kernel의 크기를 통해 Overlap Issue를 방지하자)
- Upsampling할때, 각 Layer마다 대응되는 Contracting Path의 Layer와 Concatenate해준다.
- Concatenate후 다시 채널사이즈를 줄여주는 작업을 반복한다
>
---
#### 주의점
![](https://velog.velcdn.com/images/abrahamkim98/post/1958756d-afda-4275-a7ce-1237530b3385/image.png)
>
중간에 어떤 Feature맵도 홀수의 크기를 가지면 안된다.
