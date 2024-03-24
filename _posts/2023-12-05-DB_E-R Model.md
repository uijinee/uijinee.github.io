---
title: "5. ER Modeling"
date: 2023-12-05 22:00:00 +0900
categories: ["Computer Science", "DataBase"]
tags: ["db", "database"]
use_math: true
---


앞에서는 DataBase를 사용하는 방법에 대해서 알아보았다면, 여기서는 DataBase를 어떻게 설계해야 하는지, 즉 DataBase의 Schema를 어떻게 만들지에 대한 생각을 해보는 부분이다.

이를 위해서 Entity-Relationship Model이라는 개념을 도입하게 되는데, 이 E-R Model을 먼저 만들고, 이를 통해 Relational Model을 만들게 된다.

즉, E-R Model이 무엇인지배우고, 이를통해 E-R Diagram을 만들고, 이 E-R Diagram으로부터 Relational Model을 만드는 방법을 살펴볼 것이다.

---
# E-R Model
E-R Model은 다음과 같이 DataBase를 Entity와 Relationship으로 나누어 생각하는 방법이다.

이 때, DataBase를 설계하는 단계이기 때문에 앞에서 배웠던 Relational DataBase의 속성과는 다른 점이 있을 수 있다.
(예를들어 E-R Model에서는 Atomic하지 않은 Data도 허용한다.)

따라서 일단 우선은 E-R Model이 Relational DataBase와 다르다는 것을 인지하고, 이 E-R Model이 어떻게 Relational DataBase로 변해가는지에 초점을 맞추어야 할 것 같다.

---

# Entity

![Alt text](/assets/img/post/database/entityset(1).png)

*(Entity Set: 주로 직사각형으로 표현한다.)*

## 1. Intro

### 1) 정의

> ![Alt text](/assets/img/post/database/entityset(2).png)
> 
> Entity는 위의 데이터를 실체화한 대상이다. 즉, Data에 실제로 들어가는 값들이라고 생각할 수 있다.
>
> 이때, 이 테이블의 이름을 Entity Type라고 한다. 즉, 위의 테이블은 Instructor Entity Type라고 할 수 있다.
>
> 또 이때 Entity들의 Set, 즉 Entity Type의 현재 상태를 Entity Set이라고 한다.

### 2) Attribute와 Domain
>
> 모든 Entity들은 Attribute를 가진다.
> 
> 이 Attribute에는 Domain이 존재하는데 `varchar`, `int`와 같이 구체적인 값이 아니라 추상적인 개념으로 접근해보자.
>
>
> - **Simple & Composite**<br>
>  : `( )`로 표현한다.<br>
>  : 단일 값을 갖는 데이터 `or` 여러 데이터가 구조를 이루어 형성된 하나의 데이터<br>
>  *(ex. Name(firstname, lastname), Address(Street, City, State) )*
>
> - **Single-Valued & Multi-Valued**<br>
>  : `{ }`로 표현한다<br>
>  : 단일 값을 갖는 데이터 `or` 여러 데이터를 한번에 입력할 수 있는 데이터<br>
>  *(ex. {Phone_number(Area_code, Phone_number)} )*
> 
> - **Complex**<br>
>  : Composite와 Multi-Valued가 합쳐진 유형
> 
> - **Derived**<br>
>  : 다른 Attribute를 통해 유추할 수 있는 데이터를 의미한다<br>
>  *(ex. age &rarr; birth라는 Attribute로 부터 추측이 가능함)*<br>
>  *(ex. number of employee &rarr; 직원수를 단순 카운트 해도 됨)*
>
>
> *(참고)*<br>
> *: 후에 Relational Model에서는 Simple하고 Single-Valued인 데이터만 남게된다*.
>
> ---
> **Null 값의 의미**
>
> - 알려지지 않음(모름)<br>
>  : ex. John은 핸드폰이 있으나 그의 핸드폰 번호는 알지 못함
>
> - 아직까지는 이용할 수 없음(보류)<br>
>  : ex. John은 자신의 번호를 의도적으로 적지 않음
>
> - 적용할 수 없음(정의되지 않음)<br>
>  : ex. college_degree는 대학 미 졸업자에게는 정의되지 않음

### 3) Key
>
> Relational Model과 마찬가지로 E-R Model의 EntitySet도 Key를 가진다.<br>
>  *(Primary Key, Candidate Key, Super Key)*
>
> - **Composite**<br>
>   : Key는 Composite일 수 있다.(두개 이상일 수 있다.)
>
> - `_`(밑줄) 로 표현한다.<br>
>   *(ex. Employee( <u>ID</u>, Name, Phone_number) )*<br>
>   *(ex. Room( <u>Room_number</u>, <u>Building_number</u>, area, max) )*

---
# Relationship

![Alt text](/assets/img/post/database/relationshipset(1).png)

*(Relation Set: 주로 마름모로 표현한다.)*

## 1. Intro

### 1) 정의
 
> ![Alt text](/assets/img/post/database/relationshipset(2).png)
>
> Relationship은 Entity들이 관계를 맺고 있다는 것을 의미한다.
>
> 마찬가지로, 이 Relation의 이름을 Relation Type라고 한다.<br>
> 즉, 위의 테이블은 Student_has_Instructor Relation Type이라고 할 수 있고 이 Relationship Type의 현재 상태를 Relationship Set라고 한다
>
> 흔히 착각할 수 있는 점 중 하나는 이 Relation을 두 Eitity Set에 의해 발생하는 하나의 현상이라고 이해하면 안되고, 하나의 Data라고 생각해야한다.
>
> ---
> **Recursive Relation**
>
> ![Alt text](/assets/img/post/database/recursive_relationship.png)
> 
> 경우에 따라 하나의 Entity Type이 Relation에 두번 이상 참여하는 상황이 있다.
>
> 이 경우 참여하게 되는 역할의 이름이나 대응하는 번호를 부여하여 구분한다.

### 2) 제약조건

> RelationShip은 두 Entity Set이 관계를 맺는 것이므로 Relationship에 참여하고 있는 두 Entity들로만 Identify될 수 있다.
>
> 따라서, 이 Entity들이 어떻게 Relationship에 참여하는지에 따라 제약조건이 발생한다
>
> - Cardinality<br>
>   : 관계에 참여할 수 있는 Entity의 최대 갯수
> - Participation<br>
>   : 관계에 참여하는 Entity의 최소 갯수

### 3) Attribute

> Relationship도 마찬가지로 Attribute를 가지고 이 Attribute들은 관계를 형성한 두 Entity의 Attribute로 구성된다.
>
> 이때, 기존의 Attribute 외에도 관계를 표현하는 추가적인 Attribute가 있을 수 있다.
> 
> 이 추가적인 Attribute는 두 Entity나 Relationship에 속하게 될 수 있는데, 각각의 경우는 Cardinality에 따라 가능한 경우들이 달라진다.
> 
> - `1 : 1` Relationship<br>
> ![Alt text](/assets/img/post/database/attribute-11relationship.png)<br>
>   Relationship: 가능<br>
>   1번 Entity: 가능<br>
>   2번 Entity: 가능<br>
>
> - `N : 1` Relationship<br>
> ![Alt text](/assets/img/post/database/attribute-N1relationship.png)<br>
>   Relationship: 가능<br>
>   N쪽 Entity: 가능<br>
>   1쪽 Entity: 불가능
>
> - `N : M` Relationship<br>
> ![Alt text](/assets/img/post/database/attribute-NMrelationship.png)<br>
>   Relationship: 가능<br>
>   N쪽 Entity: 불가능<br>
>   M쪽 Entity: 불가능
>
> *(참고)<br>
> 관계를 설명하기 위한 추가적인 Attribute를 위해 Relationship을 만드는 것이 아니라는 점 유의*


## 2. 특성

### 1) Cardinality

> Cardinality는 관계를 정의하는데 필요한 Entity의 Maximum Number를 의미한다.
> ![Alt text](/assets/img/post/database/cardinality.png)
>
> 즉, Many부분에서는 N개 이하의 Entity가 관계에 참여한다는 뜻이고,<br>
> One부분에서는 오직 한개의 Entity만 관계에 참여한다는 뜻이다.
> 
> ---
> **표현방법**
>
> ![Alt text](/assets/img/post/database/cardinality_expression.png)
>
> ---
> **My SQL**
>
> 다리가 많은 쪽이 `many`의 역할을 하고, 하나만 있으면 `one`의 역할을 한다.
>  
> ![Alt text](/assets/img/post/database/mysql_cardinality.png)

### 2) Participation(제약조건)

> Total Participation과 Partial Participation은 관계를 정의하는데 필요햔 Entity의 Minimum Number를 의미한다.
> 
> ![Alt text](/assets/img/post/database/participation.png)
> 
> 즉, Total Participation은 1이상의 Entity가 관계에 참여한다는 뜻이고<br>
> Partial Participation은 0이상의 Entity가 관걔에 참여한다는 뜻이다.
>
> ---
> **표현방법**
>
> ![Alt text](/assets/img/post/database/participation_expression.png)
> 
> 이중실선, 실선
>
> ---
> **My SQL**
>
> Mysql의 Workbench에서는 `Mandatory`라는 항목으로 `Participation`을 표현하고 있다.
>  의존성
> 
> ![Alt text](/assets/img/post/database/mysql_participation.png)
> 
> ---
> **Primary Key**<br>
>
> RelationShip에서 Primary Key를 정하는 방법은 다음과 같다.
> 
>  - **One to One**<br>
>  : <u>두 EntitySet의 Primary Key중 하나의 Primary Key로 선택한다.</u><br>
>  *(One to One Relationship은 한쪽의 한 Entity만 있더라도 Relationship을 정의할 수 있기 때문)*
>  - **Many to One**<br>
>  : <u>Many를 담당하는 EntitySet의 Primary Key를 Primary Key로 선택한다.</u><br>
>  *(Many to Many Relationship은 Many쪽의 Entity가 있어야 Relationship을 정의할 수 있기 때문)*
>  - **Many to Many**<br>
>  : <u>두 EntitySet의 PrimaryKey를 합쳐 Primary Key로 정한다.</u><br>
>  *(Many to Many Relationship은 양쪽 모두의 Entity가 있어야 Relationship을 정의할 수 있기 때문이다.)*


### 3) Degree

> ![Alt text](/assets/img/post/database/degree.png)

### 4. Strong EntitySet과 Weak EntitySet

> E-R Model에는 중복된 Attribute가 각 Entity에 존재하면 안된다는 규칙이 있다. 즉, 다음과 같이 중복된 Attribute가 있을 경우 한쪽을 지워주고, 이를 Relation으로 표현해야 한다.
>
> ![Alt text](/assets/img/post/database/ermodel_rule.png)
>
> 이 때, 위의 경우에서는 겹치는 Attribute가 지워지는 쪽에서 Primary key를 구성하지 않았으므로 문제가 되진 않는다. 
>
> 하지만, 만약 그렇지 않다면, 한 EntitySet에 의해 다른 EntitySet이 결정되는 관계가 되어버린다.
>
> 즉, 이렇게 다른 EntitySet에 의존하는 EntitySet을 Weak EntitySet이라고 하고 그렇지 않으면 Strong EntitySet이라고 한다.
>
> ![Alt text](/assets/img/post/database/Identifying.png)
>
> 이 때, Weak EntitySet의 Primary Key는 더이상 제역할을 하지 못한다.<br> 
> 즉, 해당하는 Strong Entity Set의 Primary Key와 합쳐져야 Primary Key의 역할을 할 수 있다.
>
> **이 Weak EntitySet의 PrimaryKey를 Partial Key혹은 discriminator라고 한다.**
>
> ---
> **표현방법**
>
> IE표기법
> MySQL표기법
> Peter-Chen 표기법
> 이중 사각형
>
> ---
> **My SQL**
> 
> Mysql의 Workbench에서는 `Identifying Relationship`이라는 항목으로 `Weak Entity`를 표현하고 있다.
> 
> ![Alt text](/assets/img/post/database/mysql_Identifying.png)

---
## 3. E-R Diagram

![Alt text](/assets/img/post/database/erdiagram.png)

