---
title: "2. Protocol Reverse Engineering"
date: 2025-09-24 10:00:00 +0900
categories: ["Domain Knowledge", "Cybersecurity"]
tags: ["pre", "protocol", "reverse engineering", "protocol reverse engineering"]
use_math: true
---

# Using Network Trace 

## Alignment Based Approach

메시지를 정렬하고 유사성 점수 계산 -> 점수를 기반으로 클러스터링 -> 메시지들의 공통점 분석 및 형식 도출

한계점: 메시지 내용의 다양성이 정렬의 품질을 저하시켜 분석하는데 문제 발생

---
## Token Based Approach


### 1) Netzob

> #### Needleman & Wunsch alignment
> 
> 생물 정보학에서 단백질/뉴클레오티드 서열을 정렬하는데 사용되는 알고리즘이다.<br>
> 두 개의 시퀀스를 최적의 방식으로 정렬하면서, 공통된 부분은 일치시키고, 차이점은 Gap("-")을 삽입하면서 정렬 Score가 최대가 되도록 만드는 전역 정렬 알고리즘
>
> 맨 오른쪽 아래가 최종 스코어
> 
>
> #### UPGMA(Unweighted Pair Group Method with Arithmetic Mean)
> 
> 가중치 없는 pair를 그룹하고 산술평균을 내는 계층적 클러스터링 알고리즘
>
> 어떤 메시지를 먼저 묶고, 어떤 순서로 클러스터를 합칠지 결정하는데 사용되는 알고리즘

### 3) [NetPlier](https://github.com/yapengye/NetPlier)

목표: Keyword Field 유추(response or request or ...)

> #### Background
> 
> 현존하는 Protocol Reverse Engineering은 실행 추적 기반(ExeT)방식과 네트워크 트레이스 기반 방식(NetT)으로 나눌 수 있다.
> 실행 추적 기반 방식은 어플리케이션에서 사용하는 입력 버퍼등을 직접 분석하여 역공학을 수행한다. 이에, 높은 정확도를 가지지만, 실행 파일들에 접근 할 수 있어야 한다는 한계가 있다. 이에 네트워크 패킷으로 부터 직접 protocol을 유추하는 방식을 사용할 수 있고, 이는 크게 다음과 같이 나뉜다.
> 
> - Alignment based method<br>
> : 정렬 알고리즘을 활용해 메시지들을 정렬하여 유사성 점수를 계산하고, 이를 기반으로 클러스터링한 후 분석하여 Format 유추<br>
> (단점: 메시지 내용이 다양하기 때문에 정렬의 품질이 저하됨)
>
> - Token based method<br>
> : 메시지를 토큰화하여 변화를 줄인 후 정렬을 수행하고, 클러스터링하여 Format 유추<br>
> (단점: 토큰을 식별하기 위한 구분자를 필요로 하고, 토큰화가 Heuristics에 기반하기 때문에 부정확함)
> 
> 즉, 네트워크를 기반으로 PRE를 수행하기 위해서는 메시지의 클러스터링이 중요하다. 이때, 가장 문제가 되는 지점이 메시지 내용이 다양하기 때문에 정렬이 제대로 되지 않는다는 점이고, 이를 보완하기 위해 토큰화를 수행하는 방식이 등장하였다. 하지만, 여기서 토큰화 또한 Heuristic에 기반하기 때문에, 토큰이 잘못 분류되는 경우가 많아 문제가 발생한다.
> 
> ---
> #### Problem
>
> - Alignemnt based clustering<br>
> ⅰ) Alignment(Needleman & Wunsch): Pairwise 정렬이기 때문에 시간 복잡도가 높고, 정렬의 결과가 좋지 않다.<br>
> ⅱ) Clustering(UPGMA): Clustering의 결과가 threshold에 민감하게 달라지기 때문에, 프로토콜마다 최적의 Threshold가 달라진다.
> 
> ![alt text](/assets/img/post/cybersecurity/alignment_based_clustring.png)
>
> ※ $m_{c_0}, m_{c_2}$가 같은 종류의 패킷(요청==82)이고, $m_{c_0}, m_{c_1}$은 다른 종류의 패킷임에도 불구하고 니들먼 브니쉬 알고리즘의 잘못된 결과로 인해 $m_{c_0}, m_{c_1}$가 같은 Cluster로 판별되는 경우가 많다.
> 
> - Token based clustering<br>
> ⅰ) Binary Protocol에 명확한 구분자(경계)가 없음<br>
> ⅱ) Binary Token $\longleftrightarrow$ Text Token 구분이 정확하지 않음<br>
> $\quad$ (binary token이 길거나, text token이 짧은 경우 등)<br>
> ⅲ) Representation Token이 정확하지 않아 Over-Clustering이 발생<br>
> $\quad$ (같은 타입의 메시지라도, 특정 위치의 바이트 값이 달라 서로 다른 타입으로 분류하는 경우)
> 
> ※ Token based 방식은 ASCII 범위 내의 byte 값이면 Text라고 간주
>
> 이러한 내용들을 정리해보면 다음 두 문제가 핵심이 된다.
> - 같은 타입의 메시지를 분류하기 위해 비슷한 값이나 패턴을 찾아 클러스터링 하는 방식을 사용한 것.
> - 기존의 방법들은 한쪽(클라이언트/서버)에서 오는 메시지 데이터만 분석한 것.
>
> ---
> #### Approach
>
> (들어오는 프로토콜이 모두 같은 프로토콜이라고 가정)
> 
> Netplier에서는 기존의 결정론적인 방법을 사용하는 대신 Posterior distribution을 계산하여 활용하여 keyword 추출하는 방식을 제안한다.
>
> 1. Preprocessing: application layer로 정형화
> - time stamp, IP, port number, data(payload) 추출
> - timestamp, IP, port number를 활용해 session number 라는 feature 부여
>
> ![alt text](/assets/img/post/cybersecurity/keyword_field_gen.png)
> 
> 2. Keyword field generation: clustering을 위한 keyword field의 후보 추출
> - 메시지 정렬(MSA, Multiple Sequence Alignment)
> - 정렬된 결과에서 각 바이트를 Unit field로 취급(가장 Conservative한 결과)
> - Unit field를 static한 부분과 dynamic한 부분 두 종류로 분류
> - 연속된 같은 종류의 unit field를 조합해 Compound field 후보 생성(keyword가 꼭 단일 바이트일 필요는 없기 때문에)
> - 이때, 클라이언트옹 필드 후보와 서버용 필드 후보를 따로 생성
>
> ※ 0 byte $<$ 후보군 크기제한 $<$ 10 byte
>
> 3. Probabilistic Keyword Identification
> 
> 

### 4) [CNNPRE](https://ieeexplore.ieee.org/document/10287339)