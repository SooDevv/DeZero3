import numpy as np


class Variable:
    def __init__(self, data):
        # input 예외처리. Variable은 ndarray, None만 받을 수 있음.
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}는 지원하지 않습니다.')
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # self.data와 형상/데이터 타입이 같으며 모든 요소는 1
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y)) # output도 ndarray 인스턴스 보장
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    # f = Square()
    # return f(x)
    return Square()(x)


def exp(x):
    return Exp()(x)


def as_array(x):
    if np.isscalar(x): # int, float -> scalar
        return np.array(x)
    return x

if __name__ == '__main__':
    x = Variable(np.array(0.5))
    # a = square(x)
    # b = exp(a)
    # y = square(b)
    y = square(exp(square(x)))

    y.backward()
    print(x.grad)