---
title: "0. Summary"
date: 2023-06-04 22:00:00 +0900
categories: ["Computer Science", "Operating System"]
tags: ["os", "operating system"]
use_math: true
---

# 운영체제

![alt text](/assets/img/post/operating_system/operating_system.png)

- **정의**: 실행할 프로그램들에 자원을 할당하고 올바르고 효율적으로 실행되도록 돕는 프로그램
- **위치**: 커널영역
- **구성요소**: 커널, UI
    - 커널: 운영체제의 핵심 부분을 담당하는 곳.
    - UI: GUI, CUI 등 운영체제의 동작에 핵심은 아님. 
- **이중 모드**: 운영체제의 동작을 보호하기 위한 기법
    - 사용자 모드: CPU의 자원 접근이 불가능한 모드
    - 커널 모드: CPU의 자원 접근이 가능한 모드<br>
    (커널 모드에 진입하기 위해서는 시스템 콜이라고 부르는 소프트웨어 인터럽트가 필요하다.)
- **커널의 역할**
    - ⅰ) 프로세스 관리: 생성/실행/삭제 관리
        - 프로세스/스레드
        - 프로세스 동기화
        - 교착상태 해결
    - ⅱ) 메모리 관리
        - CPU 스케줄링
        - 메모리(페이징, 스와핑)
    - ⅲ) 파일 시스템 관리
        - 파일/폴더 (보조 기억 장치)
    - ⅳ) 디바이스 관리

---
## 1. 프로세스

- 정의: 현재 실행중인 프로그램
- 종류
    - 포그라운드 프로세스: 사용자가 볼 수 있는 프로세스
    - 백그라운드 프로세스: 사용자가 볼 수 없는 프로세스

모든 프로세스들은 CPU가 필요하다. 이에 모든 프로세스들은 한정된 시간 만큼만 CPU를 이용하고 타이머 인터럽트가 발생하면 다음 프로세스에 양보한다. 이를 위해 다음이 필요하다.

### 1) 프로세스 구성 요소

> #### ⅰ.  프로세스 제어 블록(PCB)
>
> | ![alt text](/assets/img/post/operating_system/pcb.png)<br> ![alt text](/assets/img/post/operating_system/process_state.png) | **정의**: 프로세스를 관리하기 위해 사용하는 자료구조<br><br> **생성**: 프로세스 생성시 커널 영역에 생성, 종료시 폐기<br><br> **저장된 정보(상품의 태그와 같은 역할)**<br> - **PID**: 프로세스 식별 변호<br> - **레지스터 값**: 실행을 재게할 레지스터들의 복원 값 <br> - **프로세스 상태**: 생성, 준비, 실행, 대기(입출력), 종료<br> - **CPU 스케줄링 정보**: 프로세스가 언제 CPU를 할당 받을지<br> - **메모리 정보**: 페이지 테이블 정보<br> - ...|
>
> ---
> #### ⅱ. 문맥교환(Context Switch)
> 
> | ![alt text](/assets/img/post/operating_system/context_switch.png) | ![alt text](/assets/img/post/operating_system/context_switch2.png) |
>
> 프로세스에서 다른 프로세스로 실행순서가 넘어갈 때 발생하는 일들
> - ⅰ) 기존에 실행되던 프로세스 A는 지금까지의 중간 정보(Context)를 백업
> - ⅱ) 뒤이어 실행할 프로세스 B의 중간 정보를 복구
> 
> ---                    
> #### ⅲ. 메모리 영역
> 
| ![alt text](/assets/img/post/operating_system/memory.png)| **코드영역**<br> - CPU가 실행할 코드(text)가 담긴 부분(read-only)<br> - 정적 할당 영역<br> ---<br> **데이터 영역**<br> - 프로세스가 유지되는 동안 유지할 데이터를 저장하는 부분<br> - 정적 할당 영역<br> ---<br> **힙 영역**<br> - 프로그래머가 직접 데이터를 할당하는 부분<br> - 동적 할당 영역(낮은 주소 → 높은 주소) <br> ---<br> **스택 영역**<br> - 데이터가 일시적으로 저장되는 공간<br> - 동적 할당 영역(높은 주소 → 낮은 주소)
>
> ---
> #### ⅳ. 계층 구조 및 프로세스 생성 방식
> 
> | ![alt text](/assets/img/post/operating_system/pstree.png) | ![alt text](/assets/img/post/operating_system/process_create.png) |
> 
> - 부모/자식 프로세스로 구성
> - 프로세스 생성 기법: fork/exec
>     - fork: 나의 "복사본"을 자식 프로세스로서 생성하는 방식
>     - exec: 내가 가지고 있는 메모리 공간을 새로운 프로그램으로 덮어쓰기<br>
>     (코드/데이터 영역을 실행할 프로그램 내용으로 바꾸고 나머지는 초기화)

### 2) 쓰레드
 
| Multi Process | Multi Thread |
| --- | --- |
| ![alt text](/assets/img/post/operating_system/multi-process.png) | ![alt text](/assets/img/post/operating_system/multi-thread.png) |
| fork(), process()<br> → 데이터, 스텍을 모두 복사하여 생성하는 방식<br> $\quad$ (메모리/시간의 오버헤드 발생) | thread()<br> → 데이터 영역은 공유한채 스텍 영역만 복사하는 방식 |

> - 정의: 프로세스를 구성하는 실행 흐름의 단위
> - 특징
>     - 여러 명령어 동시 실행 가능
>     - 모든 쓰레드는 프로세스의 자원을 공유한다
> 
> ※ 프로세스간 자원을 공유하지 않지만, 자원을 주고받을 수 있는 방식(IPC)은 존재한다(파일을 통한 통신, 소켓 등)

### 3) 프로세스 동기화

| ![alt text](/assets/img/post/operating_system/mutex.png) | ![alt text](/assets/img/post/operating_system/semaphore.png) |

> - 동기화: 공동의 목적을 위해 동시에 수행되는 프로세스들의 수행 시기를 맞추는 것
>     - **경쟁조건(Racing Condition)**: 병행 프로세스가 하나의 자원을 공유하게 되어 발생하는 문제
>     - **임계영역(Critical Section)**: 프로세스 코드 영역 중 공유 자원을 접근하는 코드 영역
> 
> - 동기화 방식
>     - **실행 순서 제어를 위한 동기화**: 프로세스를 올바른 순서대로 실행하기<br>
>     EX. Reader Writer problem: 파일이 먼저 써진 후에 읽어야 하기 때문에 실행순서가 중요
> 
>     - **상호 배제를 위한 동기화**: 동시에 접근해서는 안되는 자원에 하나의 프로세스만 접근하게 하기<br>
>     EX. Bank account problem: 잔액이라고 하는 공유 자원을 다룰 때, A가 write를 완료하기 전에 B가 read-write를 새로 할 경우 발생 <br>
>     EX. Producer & Consumer problem
> 
> - 상호 배제를 위한 동기화 핵심 요소
>     - **ⅰ) 상호배제(Mutual Exclusion)**: 한 프로세스가 임계 구역에 진입했다면 다른 프로세스는 들어올 수 없다.
>     - **ⅱ) 진행(Progress)**: 임계구역에 아무도 진입하지 않았으면 프로세스에 들어갈 수 있어야 한다.
>     - **ⅲ) 유한 대기(Bounded Waiting)**: 임계구역에 들어오기 위해 무한정 대기해서는 안된다.
>  
> ---
> #### 운영체제에서의 동기화
> 
> **1. 뮤텍스 락**: 오직 1명만 자원 접근이 가능해야할 때(Critical section을 보호해야 할 때)
>     - acquire: 임계구역이 잠겨있는지 확인하고, 잠겨있지 않으면 잠구고 접근
>     - release: 잠금 해제
> 
> ```
> int number = 2000000
> pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
> 
> void *decay_number(void *t){
>     pthread_mutex_lock(&lock)
>     number -= 1
>     pthread_mutex_unlock(&lock)
> }
> ```
> 
> **2. 세마포**: 한정된 수의 자원이 있고, 그만큼만 접근을 허용할 때.(Ex. 티케팅 상황)
>     - 자원의 수가 1일 경우 뮤텍스락과 같음
>     - 자원을 0으로 두고 먼저 실행할 프로세스 뒤에 signal, 다음에 실행할 프로세스 앞에 wait를 붙이면 순서대로 동작
>     - Bank Account Problem 해결 불가
> 
> ```
> wait()
> // 임계구역
> signal()
> ```
> 
> ※ Busy waiting: lock이 풀릴 때 까지 임계구역의 상태를 확인하는 것 (Spin-lock)
> ※ PCB를 이용하는 방식: lock이 걸려있을 경우 대기상태로 만들고, 자원이 생길 경우 대기큐의 프로세스를 준비 상태로 만드는 것
> 
> **3. 모니터**: 공유 자원을 내부에 숨기고 공유 자원에 접근하기 위한 인터페이스만 제공하는 방식

### 4) 교착상태(Dead lock)

> #### ⅰ) 자원 할당 그래프
> 
> | ![alt text](/assets/img/post/operating_system/source_graph.png) | ![alt text](/assets/img/post/operating_system/source_graph2.png) |
> 
> 위의 그림 처럼 사용(검정 선), 대기(빨강 선), 자원(박스), 프로세스(원)을 각각 연결하여 표현
>
> ---
> #### ⅱ) 교착 상태 발생 조건
> 
> 1. 상호 배제: 한 프로세스가 사용하는 자원을 다른 프로세스가 사용할 수 없는 상태
> 2. 점유와 대기: 자원을 할당받은 상태에서 다른 자원을 할당 받기 기다리는 상태
> 3. 비선점 스케줄링: 어떤 프로세스도 다른 프로세스의 자원을 강제로 빼앗지 못하는 상태
> 4. 원형 대기: 프로세스들이 원의 형태로 자원을 대기하는 상태
>
>> 위 네 가지 조건 중 하나라도 만족하지 않으면 교착 상태가 발생되지 않음.<br>
>> 위 네 가지 조건을 모두 만족하면 교착 상태가 발생할 수도 있음
>
> ---
> #### ⅲ) 교착 상태 해결 방안
>
> 1. **예방** (교착 상태가 발생할 조건 중 하나를 없애버리기)
>   - 점유와 대기 삭제: 특정 프로세스에 자원을 모두 할당하거나, 아예 할당하지 않는 방식(비효율 적임)
>   - 비선점 조건 삭제: 선점이 가능한 자원(CPU)에 한해서만 효과적인 방식
>   - 원형 대기 삭제모든 자원에 번호를 붙이고, 이 자원을 오름 차순으로만 할당되도록 하는 방식(비효율 적임)
> 
> 2. **회피** (은행원 알고리즘)
>   - 교착 상태가 발생하지 않을 만큼만 자원을 배분하는 방식
>       - 안전 순서열: 교착 상태 없이 안전하게 자원을 할당할 수 있는 순서
>       - 안전 상태: 안전 순서열이 있는 상태
>       - 불안전 상태: 안전 순서열이 없는 상태
> 
> 3. **검출과 회복**
>   - 선점을 통한 회복: 프로세스가 해결될 때 까지 한 프로세스씩 자원을 몰아주는 방식
>   - 강제 종료를 통한 회복: 교착상태가 해결될 때 까지 프로세스를 강제 종료하는 것
>
> 4. **무시(타조 알고리즘)**
>   - 교착 상태는 발생하지 않을 것이라고 가정하고 대책을 취하지 않는 방식
>   - 교착 상태의 발생 빈도에 비해 해결에는 상대적으로 많은 비용이 들기 때문
>   - 현재 UNIX와 Window 등 모든 운영체제에서 사용
>   - 미사일 시스템 등 시스템 재시작이 위험한 곳에서는 적합하지 않음

---
## 2. 자원 관리

### 1) CPU 스케줄링

![alt text](/assets/img/post/operating_system/cpu_scheduling.png)

운영체제가 프로세스를 우선순위에 따라 공정하고 합리적으로 CPU 자원을 배분하는 것

EX. 입출력 작업이 많은 프로세스 수행 → CPU 작업이 많은 프로세스 수행

- 스케줄링 큐
    - 준비 큐: CPU를 이용하고싶은 프로세스들이 있는 큐
    - 대기 큐: 입출력 장치를 이용하고 싶은 프로세스들이 있는 큐

- 선점형 스케줄링(preemptive scheduling): 우선순위가 높은 프로세스가 현재 프로세스를 중단시키고 CPU를 점유하는 방식
    - 장점: 모든 프로세스가 골고루 자원을 이용한다.
    - 단점: 문맥 교환에서의 오버헤드가 많다.

- 비선점형 스케줄링(non-preemptive scheduling): 현재 프로세스가 CPU를 반환할때 까지 지속적으로 사용하는 방식
    - 장점: 문맥 교환에서의 오버헤드가 적다
    - 단점: 모든 프로세스가 골고루 자원을 이용하기 어렵다.


| ![alt text](/assets/img/post/operating_system/scheduling_queue.png) | ![alt text](/assets/img/post/operating_system/ready_queue.png) |


> 1. 선입 선처리 스케줄링 (FCFS, First Come First Served)
>     - 비선점 스케줄링
>     - 정의: 단순히 준비 큐에 삽입된 순서대로 처리하는 스케줄링
>     - 단점: 프로세스들이 기다리는 시간이 매우 길어질 수 있음
> 
> 2. 최단 작업 우선 스케줄링 (SJF, Shortest Job First)
>     - 비선점형 스케줄링
>     - 정의: CPU 사용 시간이 가장 짧은 프로세스부터 처리하는 스케줄링 방식 
> 
> 3. 라운드 로빈 스케줄링 (RR, Round Robin)
>     - 비선점형 스케줄링
>     - 정의: 정해진 시간동안만 CPU를 사용하도록 스케줄링하고 끝나지 않았다면 큐의 맨 뒤에 다시 삽입
>     - 단점: 타임 슬라이스의 크기에 따라 효과가 달라짐
> 
> 4. 최소 잔여 시간 우선 스케줄링(SRT, Shortest Remaining Time)
>     - 비선점형 스케줄링
>     - 정의: RR + SJF, 정해진 시간동안 CPU를 사용하되, 다음 CPU는 남은 작업 시간이 가장 적은 프로세스
> 
> 5. 우선순위 스케줄링(Starvation)
>     - 정의: 프로세스들에 우선순위를 부여하고 우선순위가 높은 프로세스부터 실행하는 방식<br>
>       (ex. SJF, SRT 모두 우선순위 스케줄링이다.)
>     - 단점: 우선순위가 낮은 프로세스는 실행이 안될 가능성이 있음(기아 현상)
>         - 해결방법(Aging): 오랫동안 대기한 프로세스의 우선순위를 점차 높이는 방식
> 
> 6. 다단계 큐 스케줄링(Multilevel queue 스케줄링)
>     - 정의: 우선순위 별로 준비 큐를 여러개 사용하는 스케줄링 방식(이 때, 큐별로 프로세스들이 존재)
>     - 단점: 큐간의 이동이 불가하여 기아 현상 발생 가능
> 
> 7. 다단계 피드백 큐 스케줄링
>     - 정의: 큐간의 이동이 가능한 다단계 큐 스케줄링<br>
>     (준비상태의 프로세스는 우선 순위가 높은 큐에 삽입하여, 대기상태마다 우선순위가 낮은 큐로 이동시킴)

### 2) 메모리 할당

**가상 메모리**: 프로세스 전체가 메모리 내에 올라오지 않더라도 실행이 가능하도록 하는 기법.
- RAM과 보조기억 장치의 Swap 영역으로 구성되어 있음

**※ 스와핑:** 현재 사용되지 않는 프로세스들을 보조기억장치의 일부 영역으로 쫓아내어 프로세스들이 요구하는 메모리 공간의 크기보다 큰 메모리를 사용할 수 있게 하는 방식

| ![alt text](/assets/img/post/operating_system/swaping1.png) | ![alt text](/assets/img/post/operating_system/swaping2.png) |


> #### ⅰ) 연속 메모리 할당
>
> | 연속 메모리 할당 | 메모리 조각 모음 |
> | --- | --- |
> | ![alt text](/assets/img/post/operating_system/memory_allocate.png) | ![alt text](/assets/img/post/operating_system/memory_compaction.png) |
>
> - 최초 적합(빈 공간 A): 메모리 내의 빈 공간을 순서대로 검색하다 적재할 공간을 발견하면 그 공간에 프로세스를 배치하는 방식
> - 최적 적합(빈 공간 C): 메모리 내의 빈 공간을 모두 탐색하여 적재 가능한 가장 작은 공간에 할당하는 방식
> - 최악 적합(빈 공간 B): 메모리 내의 빈 공간을 모두 탐색하여 적재 가능한 가장 큰 공간에 할당하는 방식
>
>> **문제점**
>> 
>> **1. 외부 단편화 문제**: 프로세스들이 실행되고 종료되길 반복하며 프로세스를 할당하기 어려울 만큼 작은 메모리 공간들로 인해 메모리가 낭비되는 현상
>>  
>> **2. 실제 물리 메모리보다 큰 프로세스 실행 불가**
>> 
>> ※ 임시적인 해결책, 메모리 조각 모음(memory compaction): 여기저기 흩어져 있는 빈 공간들을 하나로 모으는 방식
> 
> ---
> #### ⅱ) 페이징
>
> ![alt text](/assets/img/post/operating_system/paging.png)
>
> - 정의: 모든 프로세스를 연속적이고 크기가 일정한 논리 공간(페이지)으로 자르고, 이를 메모리의 물리 공간(프레임)에 불연속적으로 할당하는 가상 메모리 관리 기법
>
> - 특징
>   - 모든 프로세스는 페이지 테이블이 존재한다.
>   - 내부 단편화가 발생 가능하다.(마지막 페이지에서 발생하는 메모리 낭비)
>   - 프로세스를 실행하기 위해 모든 페이지가 적재될 필요는 없다.(스왑 인(페이지 인), 스왑 아웃(페이지 아웃)이 존재한다.)
> 
> ※ Segmentation: 고정된 크기를 사용하지 않고 여러 크기의 segment를 사용하는 것
> 
> ---
> ![alt text](/assets/img/post/operating_system/page_table.png)
> 
> - PTBR: 각 프로세스의 페이지 테이블이 적제된 주소를 가리키는 레지스터 
>
> - TLB: 페이지 테이블이 메모리에 있으면 메모리 접근시간 오버헤드가 발생하기 때문에 사용하는 캐시 메모리
>   - TLB hit: CPU가 접근하려는 논리주소가 TLB에 있을 때,
>   - TLB miss: CPU가 접근하려는 논리주소가 TLB에 없을 때
> 
> - 페이지 테이블
>   - 논리 주소: (페이지 번호, 변위)
>   - 페이지 테이블 엔트리
>       - 페이지 번호, 프레임 번호
>       - 유효비트: 현재 해당 페이지에 접근할 수 있는지(페이지가 메모리에 적재되어 있는지 → 적재되어 있지 않으면 **page fault** 발생)
>       - 보호비트: 읽기, 쓰기, 실행과 관련된 권한을 가지고 있는 비트
>       - 참조 비트: CPU가 해당 페이지를 방문한 적이 있는지
>       - 수정 비트: CPU가 이 페이지를 수정한 적이 있는지(스왑 아웃 시, 변경 내용을 보조 기억 장치에 쓸지 여부 결정)
>
> ---
>
> | ![alt text](/assets/img/post/operating_system/write_copy.png) | ![alt text](/assets/img/post/operating_system/hierarchical_paging.png) |
> 
> - 쓰기 시 복사: fork시 바로 복사하여 자식 프로세스를 생성하는 것이 아니라, 부모 프로세스/자식 프로세스에서 수정이 발생하는 페이지에 한해서만 복사하여 사용하는 방식
> - 계층적 페이징: 프로세스 테이블의 크기가 커질 때, 메모리 효율성을 위해 페이지 테이블을 페이징하여 여러 단계의 페이지를 두는 방식
>
> ---
> #### 페이지 교체 알고리즘
>
> - 요구페이징: 모든 페이지를 적재하지 않고 필요한 페이지만 메모리에 적재하는 기법
>   - → 명령어 실행
>   - → 해당 페이지의 유효비트가 1일 경우 바로 접근
>   - → 해당 페이지의 유효비트가 0일 경우 페이지 폴트(Page fault) 발생
>   - → 페이지 폴트(Page fault) 발생 시 페이지 교체 알고리즘에 따라 희생될 페이지를 디스크에 기록
>   - → 필요한 페이지를 메모리로 적재하고 유효비트를 1로 설정
>
>> <u> 여기서 "어떤 페이지"를 내보내어 교체할지를 결정하는 것이 중요 </u>
>
> 1. **FIFO 페이지 교체**: 메모리에 가장 먼저 올라온 페이지부터 내쫓는 방식
>   - 단점: 프로그램 실행 내내 사용될 페이지가 내쫓길 수 있음
>   - Second-chance 알고리즘: 참조비트가 1일 경우 참조비트를 0으로 바꾸고 메모리에 유지, 0일 경우에는 바로 내쫓음
> 2. **최적 페이지 교체**: 앞으로의 사용 빈도가 가장 낮은 페이지를 교체하는 알고리즘
> 3. **LRU 페이지 교체**: 가장 오래 사용되지 않은 페이지를 교체하는 알고리즘
>
> ---
> #### 프레임 할당 알고리즘
>
> ![alt text](/assets/img/post/operating_system/thrashing.png)
> 
> 쓰레싱: 프로세스가 실행되는 시간보다 페이징에 더 많은 시간을 소요해 CPU의 이용률이 저하되는 문제
> - 동시에 실행되는 프로세스가 많아진다고해서 무조건 CPU의 이용률이 향상되는건 아니다.(페이지 교체 시간 상승)
>
>> <u> 즉, 운영체제는 각 프로세스가 문제 없이 실행되기 위해서 필요한 최소한의 프레임 수를 파악해야 함 </u>
>
> 1. 균등 할당: 모든 프로세스에 동등한 프레임 할당
> 2. 비례 할당: 프로세스의 크기에 비례하여 프레임 할당
> 3. 작업 집합 모델: CPU가 "특정 시간동안" 주로 "참조한 페이지 개수" 만큼만 할당하는 방식
> 4. 페이지 폴트 빈도 기반: 페이지 폴트율에 상한선과 하한선을 정하고 그 내부 범위 안에서만 프레임을 할당하는 방식
>
> ![alt text](/assets/img/post/operating_system/frame_allocate.png)

---
## 3. 파일과 디렉토리




[출처] https://www.youtube.com/watch?v=isj4sZhoxjk

https://hyonee.tistory.com/95#google_vignette

https://suhyunsim.github.io/2023-03-14/%EC%9A%B4%EC%98%81%EC%B2%B4%EC%A0%9C-%EB%A9%B4%EC%A0%91%EC%A7%88%EB%AC%B8