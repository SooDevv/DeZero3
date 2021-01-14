# Deep learning from scratch3
> 매일 1commit 하는 것을 목표로 합니다. 

### Step1 상자로서의 변수 
- 머신러닝 시스템에서는 기본 데이터 구조를 `다차원 배열`을 사용. 
  + 다차원 배열은 숫자 등의 원소가 일정하게 모여있는 데이터 구조. 
- Variable 다차원 배열만 취급하는데. 다차원 배열은 `numpy.ndarray`이고 이는 `np.array()`로 생성할 수 있음.


### Step2 변수를 낳는 함수
- 함수(Function)는 변수 사이의 대응 관계를 정의하는 것 
  + input/output 모두 Variable 인스턴스 
- 모든 함수에 공통된 부분은 Function 클래스를 만들고, 구체적인 함수는 Function 클래스 상속 후 구현
  + `__call__` 메소드 역할은 1) 변수에서 데이터를 찾는것, 2) 계산 후 데이터를 다시 변수에 담아 반환하는 것 
  + `forward` 메소드를 구현 후, 나머지 구체적인 계산 로직은 상속 후 구현. e.g. class Square(Function): pass


### Step3 함수 연결 
- 함수를 연결할 수 있는 이유는 함수의 input/output이 모두 Variable의 인스턴스로 통일되어 있기 때문에 연속으로 적용 가능.
- 단순한 함수들을 연결해서 복잡한 식도 구현할 수 있음.


### Step4 수치미분 
- 미분이란, 변화율 or 극한으로 짧은 시간(순간)에서의 변화량
- 수치미분, 미세한 차이를 이용하여 함수의 변화량을 구하는 것
  + 중앙차분(centered difference): f(x-h)와 f(x+h)
  + 전진차분(forward differnece): f(x)와 f(x+h)
 - 단점
   + 자릿수 누락으로 결과에 오차가 포함되기 쉬움.(정확도 저하) -> 어떤 계산이냐에 따라 커질 수도 있음.
   + 계산량이 많아짐(변수 갯수에 비례) -> 딥러닝에선 현실적이지 않음.
 - 장점
   + 구현은 쉬움 
   + 역전파와 기울기 확인(gradient checking)에 사용
  
 
### Step5 역전파 이론 
- 역전파: y(중요요소)의 각 변수에 대한 미분값 
- 머신러닝은 주로 대량의 매개변수를 입력 받아 마지막에 손실 함수를 거쳐 출력(중요요소)을 냄
  즉, 손실 함수의 각 매개변수에 대한 미분을 계산 
- 순전파 vs 역전파
  - 출력에서 입력 방향으로 전파하면 한 번의 전파만으로 미분 계산 가능 
  - 입력에서 출력 방향으로도 가능하나 '입력값에 대한 미분'이기 때문에 한번에 불가


### Step6 역전파 구현 
- 미분값을 저장해야하기 때문에 Variable 클래스에 `self.grad=None` 추가 
- Function class에는 순전파 인스턴스 저장, 역전파 함수 구현 


### Step7 역전파 자동화
- Step6는 새로운 계산을 할 때마다 수동으로 역전파 코드를 작성해야하는 문제점 존재
- 위의 문제를 해결하기 위해 'Define-by-run'으로 변수와 함수를 연결하여 자동화 
  + Define by run: 데이터를 흘려보냄으로써(run) 연결이 규정된다는(define) 뜻 
- 변수와 함수를 연결하기 위해선 '관계'를 파악
  + 함수 입장에서는 변수는 입력과 출력
  + 변수 입장에서의 함수는 창조
  

### Step8 재귀에서 반복문으로 
- Step7에서 backward()를 재귀로 표현 -> 메모리 비효율 -> 반복문으로 수정