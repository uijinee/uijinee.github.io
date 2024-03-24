# EE-R Model
EE-R Model은 Enhanced Entity-Relationship Model의 약자로, 기존의 Basic E-R Model에 추가적인 Concept를 적용한 모델이다.

여기서 EE-R Model이 추가한 Concept은 다음과 같다.

- SubClasses & SuperClasses
- Specialization & Generalization
- Inheritance
- Union types(Category)

즉, 여기서는 각각의 개념들을 살펴보고 기존의 Basic 모델에서 어떤 점들이 추가되는지 알아보도록 하자.

## 1. SubClasses & SuperClasses

> SubClasses와 SuperClasses에 대한 개념은 이 EER Model에 대한 전반적인 틀을 구성한다.
>
> 이 Concept는 다음과 같이 표현 가능하다.
>
> - 하나의 큰 Entity를 Subgrouping한다.
> - 여러개의 작은 Entity들을 Grouping한다.
>
> 예를들어, Secretary, Engineer, Technician이라는 각각의 Entity Type이 있을 때, 이것들은 Employee라는 하나의 상위 개념의 Entity Type로 Grouping하는 것이 가능하다.
>
>> 즉, 어떤 Entity Type이 다른 Entity Type과 `is-a`관계에 있을 때, 각각의 Entity를 SubClasses와 SuperClasses라고 표현한다.
>
> ```
> Secretary is a Employee
> Engineer is a Employee
> Thechnician is a Employee
> ```

## 2. Specialization & Generalization

### 1) Process
> Specialization와 Generalization는 모두 SubClasses와 SuperClasses을 구성하는 방법에 대한 내용이다.
> 
> ---
> **Specialization**
> 
> ![Alt text](/assets/img/post/database/specialization.png)
>
> Top Down방식으로 SuperClass의 SubClass를 정의하는 과정이다.
> 
> 이 과정은 다음과 같이 나눌 수 있다.
> 
> 예를들어, 하나의 `Vehicle`라는 Entity를 먼저 정의한 후, `용도`나 `외형`이라는 Attribute를 기준으로 자동차, 트럭, 오토바이, 자전거 등으로 나눌 수 있다.
>
> 또한 `Employee`같은 경우는 그 `역할`이라는 Attribute를 기준으로 Manager, 일반 사원으로 나눌 수 있다.
>
>
> ---
> **Generalization**
> 
> ![Alt text](/assets/img/post/database/generalization(1).png)
> 
> Bottom Up방식으로 SubClass의 SuperClass를 정의하는 과정이다.
>
> ![Alt text](/assets/img/post/database/generalization(2).png)
> 
> 예를들어, 위와 같이 Car와 Truck이라는 Entity들이 있었을 때, 여기서 공통되는 Attribute를 합쳐 Vehicle이라는 새로운 Entity를 만들 수 있다.
>

### 2) Inheritance

> 위의 과정들의 결과 SubClass는 SuperClass의 Attribute를 상속받게 된다.
> 즉, 각 SubClass들은 SuperClass와 Predecessor SuperClass의 Attribute를 공유하면서 자신만의 새로운 Attribute를 만들게 된다.
> 
> |                    | Target        | Method    | Schema Size | Inheritance |
> |--------------------|:-------------:|:---------:|:-----------:|:-----------:|
> | **Specialization** | Group Entity  | Top-Down  | Incresed    | O (가능)    |
> | **Generalization** | Single Entity | Bottom-Up | Reduced     | X (불가능)  |
>
> 즉, Inheritance로 인해 다음이 발생한다.
> 
> - SuperClass는 단독으로 존재 가능하지만, SubClass는 반드시 SuperClass의 Member와 함께 존재한다.
>
> - Subclass의 Attribute는 Specific Attribute, Local Attribute라고 불린다.
>
> - Subclass는 다른 Entity와 relation을 이룰 수 있다.
>
> ---
> **Shared SubClass**
>
> 이때 다음과 같이 여러 SuperClass를 갖는 SubClass가 존재할 수 있다.
>
> ![Alt text](/assets/img/post/database/shared_subclass.png)
>
> 이러한 경우는 최대한 피하는게 좋다. 


### 3) Constraint

> **1. SubClass의 결정**
>
> ![Alt text](/assets/img/post/database/constraint_type.png)
> 
> - **Predicate-defined**(*= Condition-defined*)<br>
>   : 조건에 의해 Constraint가 형성될 때<br>
>   (표시: 상위클래스와 하위클래스를 연결하는 줄 옆에 조건을 기록)
>
> - **Attribute-defined**<br>
>   : 하나의 SuperClass가 갖는 Attribute와 동일한 Attribute를 갖는 SubClass들을 이 Attribute들을 기준으로 Specialization할 때
>
> - **User-defined**<br>
>   : User에 의해 Membership이 결정될 때
>
> ---
> **2. 중복**
>
> ![Alt text](/assets/img/post/database/disjoint_overlapping.png)
> 
> - **DisJoint**<br>
>   : 하나의 Entity는 반드시 하나의 SubClass에만 포함됨<br>
>   (표시: `d`)
>
> - **OverLapping**<br>
>   : 하나의 Entity는 여러개의 SubClass에 동시에 포함될 수 있음<br>
>   (표시: `o`)
>
> ---
> **3. Completeness(Exhaustiveness)**>
>
> ![Alt text](/assets/img/post/database/completeness_constraint.png)
> 
> - Total<br>
>   : SuperClass의 모든 Entity는 SubClass들에 참여해야 한다.<br>
>   (표시: `double line`)
> 
> - Partial<br>
>   : SubClass에 참여하지 않는 SuperClass도 존재가 가능하다.<br>
>   (표시: `slingle line`)
>
> ---
> *(참고)*
> 1. 다음을 구분할 수 있어야 함
>       - *DisJoint, Total*
>       - *DisJoint, Partial*
>       - *Overlapping, Total*
>       - *Overlapping, Partial*
>
> 2. Generalization의 경우 보통 Total이다.
>
> 3. [참고할만한 사이트](https://www.geeksforgeeks.org/constraints-on-generalization/)

---
## 3. Union types(Category)

> 다음과 같이 두개 이상의 SuperClass가 필요할 때 사용한다.
> 
> ![Alt text](/assets/img/post/database/uniontype.png)
> 
> 하지만 이러한 경우는 별로 존재하지 않기 때문에 1실제로는 거의 사용하지 않는다.
>


![Alt text](/assets/img/post/database/eerdiagram.png)
