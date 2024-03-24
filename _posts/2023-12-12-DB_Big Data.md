# NOSQL

RDBMS는 과거에 Transaction과 Concurrency Control 등을 통해 안전적으로 데이터를 관리할 수 있다는 장점 덕분에 많이 사용되어 왔다.

하지만 RDBMS는 한 대의 컴퓨터에서 실행되도록 설계되었기 때문에, 데이터와 트래픽의 양이 증가할수록 처리 비용이 매우 커진다는 단점이 발생하였다.

이 점을 개선하기 위해 나온 DBMS가 NOSQL이다.<br>
NOSQL은 Not Only SQL의 약자로 DBMS의 한 종류인데, NOSQL은 데이터의 일관성을 약간은 포기하는대신, 여러대의 컴퓨터에 데이터를 저장할 수 있도록 하는 것을 목표로 만들어졌다.

이에 NOSQL은 RDBMS의 "**ACID**"와는 달리 "**BASE**"라는 성질을 사용한다.
이 BASE의 내용은 다음과 같다.
- **Basically Available(가용성)**<br>
    : 데이터에는 항상 접근이 가능해야 한다. &rarr; 복사본 저장 + Sharding(DB를 Shard로 잘게 나누어 관리)
- **Soft-State(독립성)**<br>
    : 분산 노드 간 업데이트는 데이터가 노드에 도달한 시점에 갱신
- **Eventually Consistency(일관성)**<br>
    : 일정 시간 경과시 Consistency가 만족된다는 뜻으로, Lazy Consistency라고도 한다.<br>
    (즉, 언젠가는 Consistency를 만족한다.)

위의 내용을 보면 알 수 있듯이 NOSQL은 완화된 일관성을 사용하기 때문에 RDBMS의 ACID가 보장되지 않는다.

> #### CAP Theorem (브루어의 정리)
>
> ACID와 BASE를 둘 다 만족하는 DataBase를 만들 수 없냐는 의문이 들 수 있다.
>
> 이것이 불가능하다는 것을 정리한 이론이 CAP Theorem인데,<br>
> 이는 다음 3가지 조건을 모두 만족하는 분산 컴퓨터 시스템이 존재하지 않다는 것을 정리한 이론이다.<br>
>
> ![Alt text](/assets/img/post/database/cap_theorem.png)
>
> - Consistency<br>
>   : 모든 노드가 같은 순간에 같은 데이터를 볼 수 있다.
> - Availability<br>
>   : 모든 요청이 성공 또는 실패 결과를 반환할 수 있다
> - Partition Tolerance<br>
>   : 메시지 전달이 실패하거나 시스템 일부가 망가져도 시스템이 계속 동작할 수 있다.
> 


## Document-based
mongo db
## Key-Value Stores
redis
## Column-based
cassandra
## Graph-based
neo4j

--- 
# Big Data

- Hadoop
![Alt text](/assets/img/post/database/hadoop_rdbms.png)
![Alt text](/assets/img/post/database/hadoop_echosystem.png)

### 필수논문
> - The Google File System(2003)
> - MapReduce(2004)
> - Bigtable(2006)