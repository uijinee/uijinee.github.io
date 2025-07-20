---
title: "[Tech Review] 당근마켓 ML"
date: 2025-08-26 12:00:00 +0900
categories: ["Artificial Intelligence", "tech review"]
tags: ["tech review", "ml"]
use_math: true
---

## 1. [게시글 필터링](https://medium.com/daangn/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9C%BC%EB%A1%9C-%EB%8F%99%EB%84%A4%EC%83%9D%ED%99%9C-%EA%B2%8C%EC%8B%9C%EA%B8%80-%ED%95%84%ED%84%B0%EB%A7%81%ED%95%98%EA%B8%B0-263cfe4bc58d)

### 1) 목적

![alt text](/assets/img/post/tech_review/daangn_nav.png)

당근마켓의 navigation bar를 보면 "동네생활"이라는 페이지가 있다. 이 페이지는 중고거래와 별도로 동네의 정보를 얻거나, 모임을 여는 목적으로 만들어진 페이지다. 하지만, 간혹가다 거래 경험과 같은 목적에 맞지 않은 글이 올라오면 이를 필터링할 필요가 있다.

### 2) How?

BERT로 중고거래 데이터(풍부함)를 먼저 학습한 후, 동네 생활 데이터(적음) 학습<br>
※ Anomaly detection을 사용하지 않은 점이 궁금하긴 하다.

> 1. BERT를 사용해 중고거래 데이터를 사전학습.
> 
> - Text 전처리<br>
> ⅰ) tockenizer: mecab(한국어 형태소 분석기)<br> (※ wordpiece(다국어 목적)를 사용하는 것보다 더 좋은 성능을 가짐)<br>
> ⅱ) Dataset 생성: 텍스트 전체를 2개의 문장으로 쪼개어 dataset 생성<br>
> 
> - 모델 학습<br>
> ⅰ) Token embedding: 50%로 다른문장으로 바꾸기, 15%로 masking하기<br>
> ⅱ) Sentence embedding: 2개의 문장을 구분하도록 설정<br>
> ⅲ) Positional embedding: 아마 fixed(sine, cosine)을 사용한 듯 하다.<br>
> ⅳ) 학습 목표: Next Sentence Prediction, Masked Language Model
> 
> ```
> A: 나는 오늘 너무 잘생겼어
> B: 게다가 오늘 입은 옷도 너무 잘어울리는거 같아.
> 
> BERT 입력: [CLS] 나는 오늘 너무 잘생겼어 [SEP] 게다가 오늘 입은 옷도 너무 잘어울리는거 같아. [SEP]
> BERT output: [CLS] → is_next, [MASK] → predict
> ```
> 
> ---
> 2. 사전학습된 BERT를 가져와 여기에 FC Layer를 붙여 추가학습
> 
> ```python
> class BERTFINETUNE(nn.Module):
>     def __init__(self, bert: BERT, num_class):
>         super().__init__()
>         self.bert = bert
>         self.decode = nn.Linear(self.bert.hidden, num_class)
>         self.softmax = nn.LogSoftmax(dim=-1)
>         
>     def forward(self, x, segment_label):
>         x = self.bert(x, segment_label)
>         out = self.softmax(self.decode(x[:, 0]))
>         return out
> ```
> 
> - imbalance문제를 해결하기 위해 Weighted random sampler 사용<br>
> (mini-batch sampling시 카테고리마다 다른 확률로 sampling하는 방식)

---

## 2. [피드 추천](https://medium.com/daangn/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EA%B0%9C%EC%9D%B8%ED%99%94-%EC%B6%94%EC%B2%9C-1eda682c2e8c)

### 1) 목적

![alt text](/assets/img/post/tech_review/daangn_feed.png)

당근마켓에 들어가자마자 보이는 화면, 즉 첫 화면에 보이는 것들을 feed라고 한다. 인스타그램, 유튜브 에서는 이미 이 피드를 추천시스템을 사용해 개인화 시킨 것을 생각해 볼 수 있다. 당근도 사용자 개인의 취향과 관심에 맞는 물건을 위해 개인화 추천 시스템을 도입하였다.

여기서 사용할 수 있는 방법은 다음과 같았다.
- 협업 필터링(Collaborative filtering)<br>
ex. 사용자의 구매 이력, 평점 등과 비슷한 다른 user의 행동 추천
- 내용기반 필터링(Content-based filtering)<br>
ex. 내가 액션을 취한 컨텐츠와 비슷한 컨텐츠 추천
- 하이브리드

하지만 당근의 추천 시스템의 목적은 실시간으로 빠르게 학습되는 개인화 추천 시스템이다.

### 2) [How?](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)

Youtube 추천 시스템을 활용하여 실시간 추천 시스템을 구현할 수 있다.

--todo--

---
## 3. [검색 품질](https://medium.com/daangn/rag%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EA%B2%80%EC%83%89-%EC%84%9C%EB%B9%84%EC%8A%A4-%EB%A7%8C%EB%93%A4%EA%B8%B0-211930ec74a1)

### 1) 목적

당근마켓의 "동네생활"에서 동네 이웃들만 알 수 있는 신뢰도 높은 업체 정보들을 얻기 위해서는 기존의 사용자가 "ⅰ) 검색어 입력 → ⅱ) 게시글 및 댓글 확인 → ⅲ) 정보 취합"의 복잡한 과정을 거쳐야 했다. 이런 불편함을 RAG를 사용해 해결할 수 있다.

이때, 기존에 존재하는 알고리즘의 문제점은 다음과 같다.<br>
- "업체 검색" $\rightarrow$ "당근 등록 업체를 보여줌"

즉, 유저들이 추천하는 업체가 아닌, 단순 등록 업체를 보여준다는 것이 문제였다.<br>
이를 해결하기 위해서 RAG(Retrieval Augmented Generation)라는 기법을 사용할 수 있다.
- "업체 검색" $\rightarrow$ "관련 업체 정보를 가진 게시글 검색" $\rightarrow$ "업체 정보 요약" $\rightarrow$ "필터링" $\rightarrow$ "업체 추천"

### 2) How?