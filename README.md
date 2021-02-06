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


### Step9 함수를 더 편리하게 
- 개선1. 파이썬 함수로 이용하기 
  ```python
  # AS-IS
  f = Square()
  y = f(x)

  # TO-BE  
  def square(x): return Square()(x)
  y = square(x)
  ```
- 개선2. backward 메소드 간소화
  - x.grad = np.array(1.0) 안하도록 Variable backward 에서 구현
  ```
  if self.grad is None: self.grad = np.one_likes(self.data)
   ```
- 개선3. ndarray만 취급하기
  1. 입력 차원에서 개선 : isinstance로 입력값 확인
  2. 출력 차원에서 개선 : 입력값이 ndarray여도 출력값이 int, float가 나올 수 있음. 
  ```python
    def as_array(x):
      if np.isscalar(x): return np.array(x)
   
    Class Function:
              ... 
        output = Variable(as_array(y))
  ```

### Step10 테스트
- 역전파 구현 테스트: 수치미분과의 기울기 확인(gradient checking) [[numerical_diff()]](https://github.com/SooDevv/DeZero3/blob/a745ea0a30542a366033d76bb3675869f9a32299/steps/step10.py#L66)
- ndarray 인스턴스간의 값이 가까운지 판정하는 함수 [[np.allclose]](https://github.com/SooDevv/DeZero3/blob/a745ea0a30542a366033d76bb3675869f9a32299/steps/step10.py#L93)


### Step11 가변 길이 인수 (순전파 편)
- 함수의 입출력(=변수)가 하나가 아닌 여러개(data structure: List)
- list comprehension으로 구현 [[code]](https://github.com/SooDevv/DeZero3/blob/a745ea0a30542a366033d76bb3675869f9a32299/steps/step11.py#L35)


### Step12 가변 길이 인수 (개선편)
- 여러개의 입력값을 받을 수 있도록 unpacking 
   ```python
  # AS-IS
  class Function(self, inputs):
                 ...
     xs = [x.data for x in inputs]  
     ys = self.forward(xs)
  
  # TO-BE
  class Function(self, *inputs):  <- 인수들을 하나로 모아서 받을 수 있음.
                 ...
     xs = [x.data for x in inputs] 
     ys = self.forward(*xs)        <-
   ```


### Step13 가변 길이 인수 (역전파 편)
- Variable backward method 또한 변경. [[code]](https://github.com/SooDevv/DeZero3/blob/a745ea0a30542a366033d76bb3675869f9a32299/steps/step13.py#L23)


### Step14 같은 변수 반복 사용 
입력변수들이 하나의 변수 일 때 현재 코드에서 생기는 문제점.
- 문제1. 같은 변수여서 미분값을 덮어쓰는 문제 
  + `x.grad = gx if x.grad is not None else x.grad + gx` 로 해결
- 문제2. 동일한 변수를 이용하여 다른 계산 
  + `def clear_grad(): return self.data = None` 로 해결


### Step15 복잡한 계산 그래프 (이론편)
- backward 시, 
    ```python
    x = Variable()
    a = A(x)
    b, c = B(a), C(a)
    y = D(b, c)
    
    # AS-IS
    y.backward()
    D -> B or C -> A -> B or C -> A 로 호출 (A의 중복 및 A는 B와 C가 끝난 후 실행되어야함)
  
    # TO-BE
    D -> B -> C -> A (generation이는 속성을 추가해서 세대별로 관리하자)
    ```
    
### Step16 복잡한 계산 그래프 (구현편)
- Generation 도입 
  + AS-IS: f = funcs.pop() 으로 마지막 원소 추출 
  + TO-BE: 세대별(generation)로 pop
- class Variable
  + set_creator()에서 변수의 generation = function.generation + 1 
  + 즉, 함수의 output은 함수 세대+1 
- class Function 
  + generation = max([input 변수 generation])
  
  
### Step17 메모리 관리와 순환 참조 
- 파이썬 메모리 관리는 1)참조카운터와 2)GC가 관리
1. 참조카운터 증가 케이스 
  - 대입 연산자 사용 
  - 함수에 인수로 전달할 때
  - 컨테이너 타입 객체(list, tuple, class)에 추가할 때 

2. GC 
  - 순환참조 시에 GC 소환
  - 약한 참조(weakref)로 해결 가능
  

### Step18 메모리 적약 모드 
- 로직상 메모리 사용을 개선할 수 있음(2가지)

<p>

1. 역전파 시, 불필요한 미분 결과를 보관하지 않고 즉시 삭제 
   불필요한 시점? backward 과정이 끝나고 나서의 값 = output.grad 
  - 역전파를 통해 구하고 싶은 값은 말단 변수이므로 중간 과정의 변수는 필요하지 않음.
    ```python
    gx = f.backward(y.grad)   <- backward 후, gx를 구하면
    if not retain_grad:
        for y in f.outputs:    
            y().grad = None   <- backward input값이였던 y.grad는 필요없어짐!
    ```

<p>
2. '역전파가 필요없는 경우용 모드' 제공

  - 신경망의 과정은 학습(training), 추론(inference)로 나뉨
    + 학습: 함수의 입력값이 inputs을 기억해야 미분할 때 쓸 수 있음(참조 카운터 +1)
    + 추론: 위의 과정이 필요 없음.
  - Config 클래스를 이용하여 학습/추론 구별.
    + 학습은 세대설정(self.generation), 연결설정(set_creator)의 과정이 있음
  - 보다 편리하게 with문을 사용하여 모드 전환 
    ```python3
    with using_config('enable_backprop', False):    
        x = Variable(np.array(2.0))
        y = square(x)
    ```
    
   