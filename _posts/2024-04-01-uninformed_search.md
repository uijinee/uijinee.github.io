---
title: "2. Uninformed Search"
date: 2024-04-01 22:00:00 +0900
categories: ["Artificial Intelligence", "Machine Learning"]
tags: ["deeplearning", "machine learning"]
use_math: true
---

# Background

### 1) Define

> 인공지능이 문제를 해결할 때 사용할 수 있는 방법은 앞으로 취할 수 있는 행동들을 미리 시뮬레이션 해보고, 최적의 결과를 내는 행동을 찾는 것이다.
>
> 이때, Uninformed Search는 이 최적의 결과가 무엇인지 알 수 없는, 즉 목표까지 얼마나 남았는지 알 수 없는 상태(Uninformed)에서 행동을 찾는 알고리즘이다.

### 2) Terms

> #### State
>
> - 인공지능이 행동을 했을 때, 가능한 상태를 의미한다. 필요없는 Detail들을 제거하는 추상화, 간략화 과정이 필요하다.<br>
>   _(ex. 바둑을 둘 때, 내가 돌을 두고 난 상태)_
>   - **Initial State**: 초기 상태
>   - **State Space**: State들로 이루어진 집합
>   - **Search Tree**: State Space를 방향 Graph가 아닌 Tree로 표현하는 것
>   - **Graph Search**: State를 방향 Graph로 표현해 Solution을 찾는 것
>
> _(visited를 조사 O: Graph Search)_<br>
> _(visited를 조사 X: Tree Search)_
>
> _(tree like: memory $\downarrow$)_<br>
> _(graph: memory $\uparrow$)_
>
> ---
> #### Action
>
> - 현재 State에서 내가 취할 수 있는 모든 행동들을 의미한다.<br>
>   _(ex. 바둑판에 바둑돌이 없는 칸에 돌을 두는 것)_
>   - **Path**: Action들로 이루어진 집합
>   - **Solution**: Path들 중 목표까지 갈 수 있는 Path
>
> ---
> #### Performance Evaluation
> 
> | Completeness | Cost Optimality | Time Complexity | Space Complexity |
> | --- | --- | --- | --- |
> | 1. Solution을 반드시 찾는지<br> 2. Solution이 없다면 알 수 있는지 | 최적의 Solution을 찾는지 | 시간 복잡도 | 공간복잡도 |   


## 1. Uninformed Search

### 1) BFS

| | Completeness | Cost Optimality | Time Complexity | Space Complexity |
| --- | --- | --- | --- | --- |
| BFS | O | X | $O(b^d)$ | $O(b^d)$ |
| Dijkstra | O | O | $O(b^{1+\frac{C^*}{\epsilon}})$ | $O(b^{1+\frac{C^*}{\epsilon}})$ |

$b$: Branch factor<br>
$d$: solution의 depth<br>
$\epsilon$: action의 최소 Cost<br>
$C^*$: Optimal Solution의 Cost

> ![alt text](/assets/img/post/machine_learning/bfs.png)
> 
> FIFO(Firt In First Out)을 기반으로 구현하는 알고리즘이다.
> 
> ```python
> def bfs(graph, start, end):
>   frontier = queue([start])
>   visited = []
>   while frontier:
>       node = frontier.pop()
>       for child in expand(node, graph):
>           if child == end:
>               return child
>           elif not visited[child]:
>               visited.append(child)
>               frontier.append(child)
>   return False
> ```
>
> ---
> #### **Uniform-Cost Search (Dijkstra)**
>
> ![alt text](/assets/img/post/machine_learning/dijkstra.png)
>
> BFS에서 Heap과 같은 Priority Queue를 사용하고<br>
> Goal test는 원래의 BFS보다 나중에 해서 Cost Optimal을 보장해 준다.<br>
> _(단, Cost는 0보다 커야함)_
>
> ```python
> def dijkstra(graph, start, end, cost)
>   frontier = heapq([start]).with(cost)
>   visited = []
>   while frontier:
>       node = frontier.pop()
>       if node == end:
>           return node
>       for child in expand(node):
>           if not visited[child] or child.cost < visited[child].cost:
>               visited[child] = child
>               frontier.append(child)
>   return False
> ```
>

### 2) DFS

| | Completeness | Cost Optimality | Time Complexity | Space Complexity |
| --- | --- | --- | --- | --- |
| DFS | O(finite state space)<br>X(infinite state space) | X | $O(b^m)$ | $O(b^m)$(graph like)<br>$O(bm)$(tree like) |
| BackTracking | O(finite state space)<br>X(infinite state space) | X | $O(b^m)$ | $O(m)$ |
| Depth-Limited | X | X | $O(b^l)$ | $O(bl)$ |
| Iter-Deepening | O | X | $O(b^d)$ | $O(bd)$ |

$b$: Branch factor<br>
$d$: Solution의 depth<br>
$m$: Maximum depth<br>
$l$: Depth Limit

> ![alt text](/assets/img/post/machine_learning/dfs.png)
> 
> LIFO(Last In First Out)을 기반으로 구현하는 알고리즘이다.
>
> ---
> #### Back Tracking
> 
> DFS는 Node의 모든 Child를 바로 생성하여 각 Child의 조건을 검사한다.
>
> 하지만 Back Tracking은 메모리를 아끼기 위해 한번에 하나씩 생성하여 조건을 검사해간다.
> 
> ---
> #### Depth-Limited Search
> 
> 일반적인 DFS는 무한 roop에 빠질 수 있다.<br>
> 이를 막기위해 ⅰ. Cycle을 확인하고, 뻗을 수 있는 Depth에 ⅱ. Limit을 설정하여 그 이상은 Cutoff한다.
> 
> ```python
> def DLS(graph, start, end, limit):
>   frontier = stack([start])
>   cutoff = False
>   while frontier:
>       node = frontier.pop()
>       if node == end:
>           return node
>       if node.depth() > limit:
>           cutoff = True
>       elif not cycle(node):
>           for child in expand(graph, node):
>               frontier.append(child)
>   return cutoff
> ```
>
> ---
> #### Iterative Deepening Search
>
> Depth Limit을 0부터 $\infty$까지 증가해가며 Depth Limited Search를 진행하는 방식
>
> ---
> #### Iterative Lengthning Search
>
> Cost Limit를 0부터 $\infty$까지 증가시키며 Search하는 방식
>
> 이때, 다음 Cost는 연속적이기 때문에 다음 Iter의 Limit은 이전 Iter의 최솟값으로 한다.
>
> ---
> #### Hybrid Approach
>
> DFS는 시간이 오래걸린다.<br>
> 따라서 BFS로 메모리가 어느정도 찰 때 까지는 탐색하다가 Iterative Deepening으로 바꾸는 전략을 쓸 수도 있다.


### 3) Bidrectional Search

| | Completeness | Cost Optimality | Time Complexity | Space Complexity |
| --- | --- | --- | --- | --- |
| Bidirectinal | O | X | $O(b^\frac{d}{2})$ | $O(b^\frac{d}{2})$ |


> ![alt text](/assets/img/post/machine_learning/bidirectional_search.png)
>
> Goal State를 알 때 쓸 수 있는 방법으로 Goal지점과 Start지점에서 모두 Dijkstra 방법으로 Search를 시작하는 방법이다.
>
> ```python
> def bid(graph_f, cost_f, graph_b, cost_b):
>   frontier_f = heapq(front).with(cost_f)
>   frontier_b = heapq(back).with(cost_b)
> 
>   visited_f = []
>   visited_b = []
> 
>   solution = False
>
>   while not Terminated:
>       if cost_f(frontier_f.top()) < cost_b(frontier_b.top()):
>           solution = proceed(graph_f, cost_f, frontier_f, visited_f, visited_b)
>       else:
>           solution = proceed(graph_b, cost_b, frontier_b, visited_f, visited_b)
> 
> def proceed(graph, cost, frontier, visited_1, visited_2):
>   node = frontier.pop()
>   for child in expand(graph, node):
>       if not visited_1[child] or child.cost < visited_1[child].cost:
>           visited_1[child] = child
>           frontier.append(child)
>           if visited_2[child]:
>               solution = join(child, visited_2[child])
>   return solution
> ```
>
> _(이때, 먼저 연결되었다고 해서 Cost Optimal이라는 보장이 없다.)_

---
## 2. Informed Search 