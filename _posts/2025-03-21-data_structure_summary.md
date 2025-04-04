---
title: "Summary"
date: 2025-03-21 12:00:00 +0900
categories: ["Computer Science", "Data Structure"]
tags: ["cs", "data structure"]
use_math: true
---
# 자료구조 핵심 정리

## 1. 배열 및 기본 자료구조

### 1) 배열(Array)

- **개념**: 동일한 자료형의 데이터를 메모리상 연속적으로 저장한 형태.
- **특징**: 인덱스를 통해 빠르게 접근 가능.
- **시간복잡도**:
  - 접근: \$O(1)\$
  - 중간 삽입/삭제: \$O(n)\$
  - 끝부분 삽입: \$O(1)\$

### 2) 연결 리스트(Linked List)

- **개념**: 노드(Node)에 데이터와 다음 노드의 주소(포인터)를 저장하여 연결한 형태.
- **특징**: 메모리가 연속적이지 않아 동적 크기 조정에 용이함.
- **시간복잡도**:
  - 접근: \$O(n)\$
  - 삽입/삭제: \$O(1)\$ (중간 지점에서 주소를 알 때)

### 3) 스택(Stack)

- **개념**: LIFO(Last In First Out), 가장 최근에 들어온 데이터가 먼저 나가는 구조.
- **주요 연산**: push, pop, peek
- **시간복잡도**:
  - 모든 연산: \$O(1)\$

### 4) 큐(Queue)

- **개념**: FIFO(First In First Out), 가장 먼저 들어온 데이터가 먼저 나가는 구조.
- **주요 연산**: enqueue, dequeue, peek
- **시간복잡도**:
  - 모든 연산: \$O(1)\$

---

## 2️. 트리(Tree)

### 1) 최소 신장 트리(Minimum Spanning Tree, MST)

- **개념**: 모든 노드를 최소 비용의 간선으로 연결한 트리.
- **주요 알고리즘**:
  - 크루스칼(Kruskal)
  - 프림(Prim)

### 2) 이진 트리(Binary Tree)

- **개념**: 각 노드가 최대 2개의 자식을 갖는 트리 구조.
- **순회 방법**:
  - 전위(Pre-order): 루트 → 왼쪽 → 오른쪽
  - 중위(In-order): 왼쪽 → 루트 → 오른쪽
  - 후위(Post-order): 왼쪽 → 오른쪽 → 루트

```python
# 전위 순회(Pre-order)
def preorder(node):
    if node:
        print(node.value)
        preorder(node.left)
        preorder(node.right)

# 중위 순회(In-order)
def inorder(node):
    if node:
        inorder(node.left)
        print(node.value)
        inorder(node.right)

# 후위 순회(Post-order)
def postorder(node):
    if node:
        postorder(node.left)
        postorder(node.right)
        print(node.value)
```

### 3) 이진 탐색 트리(Binary Search Tree, BST)

- **개념**: 왼쪽 자식 < 부모 노드 < 오른쪽 자식이라는 조건을 만족하는 이진 트리.
- **활용**: 데이터 검색에 효과적
- **시간복잡도**:
  - 탐색, 삽입, 삭제 평균: \$O(\log n)\$
  - 최악: \$O(n)\$ (편향된 트리일 경우)

### 4) 완전 이진 트리(Complete Binary Tree)

- **개념**: 모든 레벨이 꽉 차 있고 마지막 레벨의 노드들은 왼쪽부터 채워짐.
- **활용**: 힙(heap) 자료구조로 많이 사용됨
- **height(높이)**: 노드의 개수를 n이라 할 때, $\text{height} = \lfloor \log_2 n \rfloor$
- **depth(깊이)**: 루트 노드의 depth는 0이며, 특정 노드의 depth는 루트부터 해당 노드까지의 간선 개수와 같음. 배열 형태로 구현된 완전 이진트리에서는 노드의 인덱스를 i라고 할 때 depth = $\lfloor \log_2 i \rfloor$
 
### 5) 힙(Heap)

- **개념**: 완전 이진트리를 기본으로 하는 자료구조.
- **종류**:
  - 최대 힙(Max Heap): 부모 노드 ≥ 자식 노드
  - 최소 힙(Min Heap): 부모 노드 ≤ 자식 노드
- **활용**: 우선순위 큐, 힙 정렬
- **시간복잡도**:
  - 삽입, 삭제: \$O(\log n)\$
  - 최대값/최소값 접근: \$O(1)\$

---

## 3. 기타 자료구조

### 1) 그래프(Graph)

- **개념**: 정점(Vertex)과 간선(Edge)의 집합으로 표현된 자료구조.
- **표현 방법**:
  - 인접 행렬(Adjacency Matrix)
  - 인접 리스트(Adjacency List)
- **주요 알고리즘**:
  - 탐색: DFS, BFS
  - 최단 경로: 다익스트라(Dijkstra), 벨만-포드(Bellman-Ford)

### 2) 해시 테이블(Hash Table)

- **개념**: 키(key)를 해시 함수(Hash Function)로 계산하여 데이터(value)를 저장하는 구조.
- **특징**: 빠른 검색 속도 제공, 충돌(Collision) 해결 필요.
- **충돌 해결법**:
  - 체이닝(Chaining)
  - 오픈 어드레싱(Open Addressing)
- **시간복잡도**:
  - 평균 탐색, 삽입, 삭제: \$O(1)\$
  - 최악 탐색, 삽입, 삭제: \$O(n)\$