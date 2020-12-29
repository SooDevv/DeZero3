import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


def numerical_diff(f, x, eps=1e-4):
    """centered difference"""
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def funcs(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


if __name__ == '__main__':
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print('[test1] dy: ', dy)

    x1 = Variable(np.array(0.5))
    dy1 = numerical_diff(funcs, x1)
    print('[test2] dy: ', dy1)

