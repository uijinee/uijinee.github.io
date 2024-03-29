---
title: "2. Perception"
date: 2024-03-25 09:00:00 +0900
categories: ["Autonomous Driving", "Concept"]
tags: ["autonomous driving", "perception"]
use_math: true
---

# 인지 센서

인지에 주로 사용되는 센서는 다음과 같다.

|      | 카메라 | 레이더 | 라이다 |
|------|-------|--------|-------|
| 특징 | ADAS에 가장 많이 사용| 전자기파(RF신호) 의 반사파 분석| 고출력 펄스레이저의 반사신호에 대한 시간차 분석|
| 장점 | 높은 해상도<br>저렴한 가격 | 높은 정확도<br>저렴한 가격<br>강인한 성능 | 높은 정확도(먼 거리)<br>강인한 성능
| 단점 | 환경변화에 취약<br> 높은 계산량<br> 낮은 정확도(속도 및 거리)| 높은 오탐률(클러터 현상)<br>낮은 정확도(횡방향 정보)| 비싼 가격<br> 습기에 취약|

이 센서들은 차량 주변 환경에 대한 이해를 담당하게 된다.<br>
이 과정에서 센서들은 다음의 2가지 객체에 대한 <u>1) 위치, 2) 종류, 3) 상대속도</u> 등을 식별한다.

ⅰ. 동적환경객체: 차량, 사람, 동물 등<br>
ⅱ. 정적환경객체: 과속방지턱, 표지판, 신호등 등

> **Dificulty**
>
> 인지부분에서 가장 어려운 점은 근거리에서 100%의 정확도가 보장되어야 할 정도로 높은 정확도가 핵심이라는 것이다.
>
> 하지만 이는 다음과 같은 이유로 인해 매우 어렵다.
>
> - 환경적 다양성
> - 차량과 보행자의 행동 예측
> - 교통 환경의 문맥적 의미 파악 필요
> - 새로운 객체와 물체의 지속적인 업데이트
> 
> 또한 무엇보다도 이를 위해 여러 인지기술을 동시에 수행할 수 있는 하드웨어가 필요하다는 점도 인지 기술이 극복해야 하는 부분이다.

---
## 1. 카메라 
### 1) 요소
![alt text](/assets/img/post/autonomous_driving/camera_part.png)

> 카메라는 빛을 CCD나 CMOS같은 센서를 통해 감지하는 장치이다.<br>
> 주된 구성은 "렌즈", "조리개", "셔터", "몸체"이다.
>
> ---
> #### Notation
> 
> 1. 초점거리(Focal Length)<br>
>   : 렌즈부터 영상이 맺히는 센서사이의 거리
>
> 2. 화각(Field of View, FOV)<br>
>   : 화면을 구성하는 각도
>
> 3. 해상도(Resolution)<br>
>   : 2차원 배열의 크기
>
> 4. Frame Rate<br>
>   : 1초동안 보여주는 이미지의 수
>
> 5. 색상표현<br>
>   : RGB, YIQ, CMY, HIS, ...
> 
> ---
> #### 초점거리와 화각
>
> |     | 망원렌즈 | 표준렌즈 | 광각렌즈 | 어안렌즈 |
> | --- | --- | --- | --- | --- |
> | 화각 | $\sim 40^o$ | $40^o \sim 60^o$ | $60^o \sim 80^o$| $ 180^o \sim$ |
> | FOV | $70\sim 200(mm)$ | $35\sim 38(mm)$ | $15\sim 35(mm)$ | $7\sim 15(mm)$
>
>   - $Focal \, Length\propto \frac{1}{FOV}$
>   - 렌즈에 의한 외곡보정이 필요하다.


### 2) 캘리브레이션(보정)

![alt text](/assets/img/post/autonomous_driving/camera_coordination.png)

> 카메라의 2D이미지를 실제 세계에 Mapping하기 위해서는 보정작업이 필요하다.<br>
> 따라서 캘리브레이션은 2개의 카메라를 활용해 입체적인 촬영을 하는 스테레오 카메라에서 중요하게 사용된다.
>
> --- 
> #### 카메라 좌표계
>
> |   | 카메라 좌표계 |
> |---| --- |
> | 원점 | 카메라의 초점 |
> | x축 | 카메라의 오른쪽 |
> | y축 | 카메라의 아래쪽 |
> | z축 | 카메라의 정면 |
>
> ---
> #### 내부 파라미터
>
> 1. 정의<br>
>   카메라 내부의 기계적인 셋팅을 알아야 한다.
> 2. 요소<br>
>   ⅰ. **초점거리**: 렌즈의 중심과 CCD/CMOS와의 거리<br>
>   ⅱ. **주점**: 렌즈의 중심에서 이미지 센서에 수직으로 내린 점의 영상좌표(픽셀)<br>
>   ⅲ. **비대칭 계수**: 이미지 센서의 Y축이 기울어진 정도<br>
>
> ---
> #### 방법
>
> 체커보드 촬영 $\rightarrow$ 코너점 검출 $\rightarrow$ 내부 파라미터 계산

### 3) 검출

> #### Detection
> 
> 1. **One Stage Detector**<br>
> ![alt text](/assets/img/post/autonomous_driving/one-stage_detector.png)<br>
> : Classification과 Regional Proposal을 동시에 수행하는 방법.
> 2. **Two-stage Detector**<br>
> ![alt text](/assets/img/post/autonomous_driving/two-stage_detector.png)<br>
> : Classification과 Regional Proposal을 순차적으로 수행하여 결과를 얻는 방법
>   
> ---
> #### Segmentation
> 
> ![alt text](/assets/img/post/autonomous_driving/segmentation.png)
>
> 픽셀별로 물체와 배경을 분류하는 기술

### 4) Tracking

![alt text](/assets/img/post/autonomous_driving/camera_tracking.png)

> Tracking이란 검출 정보를 시간적으로 연결하여 물체별로 ID를 부여하고 추적하는 기술이다.
> 
> 과거에는 Kalman Filter와 같은 기술을 활용하여 이를 수행했지만,<br>
> 최근에는 딥러닝의 부상으로 GNN이라는 기술이 주목받고 있다.

---
## 2. 레이더

### 1) 종류

| 구분 방식 | 주파수 대역 | 송신 방식 |
| --- | --- | --- |
| 종류 | ⅰ. 24GHz: 근거리용 레이더(SRR)<br>ⅱ. 77GHz: 원거리용 레이더(LRR) | ⅰ. 펄스 레이더<br>ⅱ. 연속파 레이더 |


> #### 펄스 레이더
> 
> 1. 정의<br>
> : 1nm의 짧은 펄스를 송신 및 전파지연시간 측정하는 방식
> 
> 2. 검출 과정<br>
> ⅰ. 주기적으로 펄스를 반송파에 실어 물체 방향으로 송신<br>
> ⅱ. 수신시 지연시간 측정<br>
> ⅲ. 상대거리 예측
>
> 3. 특징<br>
> ⅰ. 펄스의 간격 $\downarrow$ = 거리분해능 $\uparrow$<br>
> ⅱ. 같은 펄스를 사용해야 하기 때문에 하나의 안테나만 사용(송수전환기)<br> 
>   $\rightarrow$ 다음 펄스의 송신 전에 반드시 반사펄스가 들어와야 함
>
> ---
> #### 연속파 레이더
>
> 1. 정의<br>
>   : 시간에 따라 주파수가 변하는 신호를 휴지시간 없이 송신 및 전파지연시간 측정 
> 
> 2. FMCW레이더(주파수 변조 연속파) 구조<br>
>  ![alt text](/assets/img/post/autonomous_driving/fmcwradar.png)<br>
>   믹서: 디처핑을 통해 두 신호의 주파수합-차를 주파수로 갖는 각각의 신호를 발생시킴
> 
> 3. 특징<br>
> ⅰ. 송수신 안테나가 분리<br>
> ⅱ. 비트 주파수를 사용해 샘필링시 하드웨어 비용이 절감됨
> 

### 2) 물체 검출

![alt text](/assets/img/post/autonomous_driving/radar.png)

> #### 검출 과정
> 
> 1. "시간-주파수" 형태의 신호 획득<br>
>   : 수신 신호에 시간에 따라 이동하는 윈도우를 적용해 주파수 성분 변환을 수행한다.
>
> 2. "거리-도플러" 영역으로 환산<br>
>   각 셀에서 신호의 세기를 Threshold를 기준으로 나눔<br>
>   ⅰ. Threshold $\downarrow$: 물체가 아님에도 물체라고 검출할 확률 상승<br>
>   ⅱ. Threshold $\uparrow$: 물체가 있음에도 검출하지 못할 확률 상승
>
> 3. 에너지 관찰<br>
>   : 하나의 셀에 대해 에너지를 보면 물체까지의 거리와 상대속도를 알 수 있다.
>
> ----
> #### CFAR(Constant False Alarm Rate)검출기
>
> 위의 검출과정에서 보면 알 수 있듯이 일정한 Threshold를 통해 검출을 할 경우<br>
> 잡음이나 **클러터**와 같은 방해신호 때문에 일정한 오탐률을 보장할 수 없다.<br>
> *(클러터: 잘못 반사된 신호에 의해 물체가 없는 위치에 신호가 검출되는 것)*
> 
> 즉, 이 Threshold를 적응적으로 조절하는 것이 필요하다
>
> - 셀-평균 CFAR검출<br>
> ![alt text](/assets/img/post/autonomous_driving/cfar.png)<br>
>  : 검출하고자하는 셀의 주변 셀의 정보를 이용해 잡음을 계산한다.
>
> ---
> #### 횡방향 각도 검출
>
> ![alt text](/assets/img/post/autonomous_driving/arrayantenna.png)
>
> 1개의 RADAR는 횡방향에 존재하는 물체에 대한 검출 성능이 떨어진다.<br>
> 이는 배열안테나를 활용해 수신된 내용의 위상 차이를 이용하면 검출할 수 있다..

---
## 3. 라이다

### 1) 종류

| 구분 방식 | 회전 유무 | 구현 방식 |
| --- | --- | --- |
| 종류 | ⅰ. 회전형 라이다<br>ⅱ. 고정형 라이다 | ⅰ. 기계식 라이다<br>ⅱ. MEMS 라이다<br>ⅲ. 플래시 라이다<br>ⅳ. FMCW 라이다 |

> 라이다는 직진성이 강한 고출력 펄스레이저 송신하고 이를 수신하여 지연시간을 분석해 물체를 탐지하고 거리를 측정하는 장치이다.
>
> 이때, 수직방향으로 동시에 송신하는 레이저 빔의 수, 즉 채널수에 따라 Resolution이 달라진다.<br>
> (보통 4채널, 16채널, 32채널, 64채널, 128채널의 제품을 사용)
> 
> ---
> #### 1. 종류별 장단점
>
> | | 회전형 라이다 | 고정형 라이다 |
> | --- | --- | --- |
> | 탐지방법 | 센서를 기계적으로 회전하여 넓은 각도의 환경정보 획득<br> | 환경정보를 획득하고자 하는 각도에 설치$\cdot$운용 | 
> | 장점 | $360^o$의 전방위 데이터 획들 가능 | 단순한 구성<br>저렴한 가격 |
> | 단점 | 복잡한 구성<br> 비싼 가격<br> 약한 내구성| 화각이 존재함|
>
> ---
> #### 2. 종류별 탐지 방법
> 
> | | 탐지방법 |
> | --- | --- |
> | **기계식 라이다** | 기계적인 모터를 사용해 탐지<br>*(현재 가장 많이 사용)* |
> | **MEMS 라이다** | MEMS기술로 사용해 작은 반사거울을 제어해 탐지<br>*(MEMS: 나노기술을 사용해 제작되는 매우 작은 기계)* |
> | **플래시 라이다** | 송신: 단일레이저 빔을 광시야각으로 확장하여 한번에 송신<br>수신: 다중배열 수신 소자를 통해 반사된 레이저 빔을 수신  |
> | **FMCW라이다** | 연속적인 신호를 보내고 분석하여 검출하는 방식(FMCW레이더와 비슷)<br> $\rightarrow$ 거리뿐만 아닌 속도 측정도 가능|


### 2) 검출

![alt text](/assets/img/post/autonomous_driving/lidar.png)

> 라이다는 고출력 펄스레이저로 인식하기 때문에 먼 거리의 물체도 정확히 감지가 가능하고 정확도도 매우 높다<br>
>
> 이때, 인식 결과는 4차원 포인트 클라우드이다.<br>
> $\rightarrow$ (x, y, z, intensity)
>
> ---
> #### 데이터 전처리
> 
> 1. 복셀기반 전처리<br>
>   : 3차원 공간을 복셀이라고 하는 작은 블럭으로 나눈 후 각 복셀마다 물체에 대한 정보를 추출하는 방식
>  
> 2. 직접 처리<br>
>   : 포인트넷과 같은 딥러닝 방법을 적용해 포인트 클라우드로부터 물체를 직접 추출하는 방식
>
> ---
> #### 검출결과 표현 방법
> 
> 1. 3차원 박스 표현
>
> 2. 조감도(Bird Eye View) 표현
>  

### 3) Tracking

![alt text](/assets/img/post/autonomous_driving/lidar_tracking.png)

> Tracking이란 검출 정보를 시간적으로 연결하여 물체별로 ID를 부여하고 추적하는 기술이다.
> 
> 과거에는 Kalman Filter와 같은 기술을 활용하여 이를 수행했지만,<br>
> 최근에는 딥러닝의 부상으로 GNN이라는 기술이 주목받고 있다.

---
## 4. 센서융합

### 1) 종류

| 구분 방식 | 융합 위치 | 센서 |
| --- | --- | --- |
| 종류 | ⅰ. 초기융합<br>ⅱ. 후기융합<br>ⅲ. 중기융합 | ⅰ. 카메라 + 라이다<br>ⅱ. 카메라 + 레이더<br>ⅲ. 카메라 + 레이더 + 라이다 |

> 보다 정확한 인지 결과를 위해서는 센서융합이 필요하다.
>
> 하지만 카메라(2차원, 카메라 좌표계), 레이더(3차원), 라이다(3차원)는 모두 사용하는 좌표계가 다르기 때문에<br>
> 이를 캘리브레이션해야 하기 때문에 어려운 과정이 수반된다.
> 
> ---
> #### 1. 센서 융합 위치
>
> | | 융합 방법 | 장점 | 단점 |
> | --- | --- | --- | --- |
> | **초기융합** | 센서의 원시데이터 정보를 융합합하는 방식| 낮은 계산량| 카메라+라이다처럼 데이터의 분포가 완전히 다를 경우 성능이 낮아진다. |
> | **후기융합** | 센서별로 인지결과를 내고 이를 마지막에 융합하는 방식 | 센서별 오작동 분석 가능 | 높은 계산량|
> | **중기융합** | 센서별로 인지결과를 내고 융합한 후,<br> 이 결과를 다시 인지에 사용  | - | - |
>
> *(딥러닝에서는 주로 중기융합을 활용한다.)*
>
> ---
> #### 2. 융합 센서 종류
> 
> | | 융합방법 |
> | --- | --- |
> | **카메라 + 라이다** | $\because$ 라이다는 해상도에 따라 검출 성능이 좌우된다.<br>ⅰ. 라이다의 해상도가 높지 않은 경우<br> $\rightarrow$ 카메라 중심의 융합기술<br>ⅱ. 라이다의 해상도가 높은 경우<br> $\rightarrow$ 라이다 중심의 융합기술  |
> | **카메라 + 레이더** | $\because$ 카메라: 인식결과 $\uparrow$ , 위치측정능력: $\downarrow$ <br> $\because$ 레이더: 인식결과 $\downarrow$ , 위치측정능력: $\uparrow$<br> $\rightarrow$ 상호 보완이 됨 |
> | **카메라 + 레이더 + 라이다** | 많은 계산량이 필요하기 때문에<br> 그만큼 하드웨어능력과 알고리즘의 구현이 필요 |
