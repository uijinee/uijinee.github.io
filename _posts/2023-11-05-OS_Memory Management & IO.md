---
title: "5. Memory Management"
date: 2023-11-05 22:00:00 +0900
categories: ["Computer Science", "Operating System"]
tags: ["os", "operating system"]
use_math: true
---

memory management와 file system비교

|                 | file system | memory management |
|-----------------|:-----------:|:-----------------:|
| Space Efficient | Disk Block  |       Page        |
| Time Efficient  |    Inode    |    Page Table     |

File & Disk = sequence of blocks
process = sequence of page
memory = sequence of frames

즉 process는 page로 나누고 memory는 frame으로 나누고 page를 empty frame에 저장<br>
후 이를 page table에 저장

1 page의 크기 = 4KiB = 0x1000 Byte
ex. main: 80483b4 -> page=8048, offset=3b4


Process란?
- Code + data + stack <br>
    \+ library Code + library Data + library Stack
- 4GB

-> process image에서 확인 가능
-> /proc/pid/maps에 있음


주의

![Alt text](/assets/img/post/operating_system/page_fault(1).png)

프로그램에서 나오는 변수의 주소는 컴파일러가 만든 process image상에서의 virtual address이다. (physical address가 아님)
&rarr; 실제 주소는 runtime시 결정되기 때문에 실행할 때마다 달라진다.
&rarr; 이는 page table에 mapping되어 저장되어 있다.

내가 보는 주소 = virtual address: process image
실제 physical 주소 = page가 어떤 frame에 들어가는 지에 따라 결정되기 때문에 달라짐

![Alt text](/assets/img/post/operating_system/page_fault(2).png)
-> 이 때문에 2 process를 실행하면 virtual address는 보통 비슷하다.
-> 하지만 이들이 memory의 frame에 들어가야 하기 때문에 동시에 수행이 가능하다

---
문제점
1. process가 너무 큼
    - 작은 memory에 process를 넣어야 함
2. Page Table이 너무 큼
3. Address mapping이 너무 커져서 느려짐

---
해결책(1)

1024개의 process가 돌아가면 4g * 1024 = 4T의 memory 필요

- demanding paging
    : 필요할 때만 page를 memory에 가지고 옴
    : 아직 memory에 없는 page를 memory로 가지고 오는 것 = page fault

- page fault 문제
    : page fault 는 intterupt
    - lru 알고리즘 사용(least recently Used를 kick out!, 오랫동안 안쓴 page를 쫓아냄)
        + swap space
        - ex 쫓아내는 page가 code이면 그냥 없앰
        - 동적으로 생성된 data가 있는 page => swap space라는 partition에 따로 저장해 놓음

- page fault로 인해 발생하는 문제들(:process의 page가 disk, memory, swap space중 어디에 있는지)
    - vma list(virtual memory area) 라는 것으로 page의 location을 기록

즉 page fault가 자주 발생하지 않도록 해야함
