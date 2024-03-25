---
title: "2. Process"
date: 2023-11-02 22:00:00 +0900
categories: ["Computer Science", "Operating System"]
tags: ["os", "operating system"]
use_math: true
---

# Process

## 1. Preview
### 1) What is a Process

![Alt text](/assets/img/post/operating_system/process.png)

> **정의**
>
> Process는 Memory에 load된 Program이다.<br>
> 이 Process는 Process Body와 Process Descriptor로 이루어져 있다.
>
> *(참고)<br>
> linux는 제일 처음 메모리에 로딩되는 Process이다.*
>
> ---
> **Process Body**
>
> - Code<br>
>   : 실제 컴파일된 코드가 복사되어 올라가는 메모리 부분<br>
>   : `system call`, `Interrupt`처리 코드
>
> - Data<br>
>   : Process실행 도중 사용하는 데이터 공간<br>
>   : `PCB`(Process Controll Block), `전역변수`, `Static변수`
>
> - Heap<br>
>   : 프로그래머가 필요할 때 사용하는 공간(런타임 시에 사용여부가 결정됨)<br>
>   : `동적할당`
>
> - Stack<br>
>   : 함수의 실행을 마치고 복귀할 주소 및 함수가 사용하는 데이터 저장<br>
>   : `지역변수`, `매개변수`, `Return값`
>
> ---
> **Process Descriptor**
>
> Process를 Control하기 위해서 운영체제가 구조체를 만들어 관리함
> 
>> Memory에 존재하는 모든 Process는<br>
>> 해당 <u>**Process Body를 가리키는 Process Descriptor가 항상 linux kernel에 존재한다.**</u>
>
> ---


## 2. Process Descriptor

### 1) Process Descriptor

> - Process당 반드시 하나의 Process Descriptor를 가지고 있음
> - linux에서는 `include/linux/sched.h`에 다음과 같이 구현되어 있다.
>       : task_struct{} -> pid, state, time quantom, mmm
>       : Kernel Mode Stack(KMS) -> eflag, cs, eip등 (process stop당시 register값)
>       : thread_union -> task_struct{} + KMS
> - 또 linux에서는 
> - linux에서 `current`라는 Pointer는 항상 Current Process를 가리키는 Process Descriptor이다.
> - linux kernel의 Process Descriptor는 `init_task`로 `arch/x86/kernel/init_task.c`와 `include/linux/init_task.h`에 구현되어 있다.
>
> ---
> **`task_struct{}`**
> 
> - time quantom: default는 보통 10ticks(10ms)로 제한, 이 time quantom시간동안 하나의 Process를 수행한 후에 다른 Process로 넘어가는식으로, 번갈아가면서 수행함 (==time sharing)
>
> - 이때 scheduler는 다음 Process로 넘어가도록 schedule하는 경우는 다음과 같다.
>   : timer expired
>   : 현재 Process가 당장 Process를 진행할 수 없고 특정 event를 기다려야만 할 때(==block당했을 때), ex. scanf() or sleep() 
>
>
> x를 schedule한다 = current가 x를 가리킨다.

### 2) Process Queue:

> Process Descriptor는 linked list로 연결되어 있음
> 이 linked list를 process queue라고 함
>
> init_task변수: 첫번째 process를 가리키는 광역 변수

### 3) Run Queue

> 모든 Process를 연결한 Process Queue와 다르게 
> block되어 있는 것을 뺀, 항상 ready 상태인 Process를 연결한 linked list
>
>
> linux는 cpu_idle()에서 항상 ready 상태로 대기하게 됨
> 이때 우선권이 낮은상태로 대기하기 때문에 다른 process들이 모두 block이 되었을 때만 선택되게 된다.
> 
> current포인터: 현재 process를 가리키는 광역 변수

---
## 3. Process Scheduling
ppt - p8