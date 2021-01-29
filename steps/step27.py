import math
import numpy as np
from dezero import Function
from dezero import plot_dot_graph


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=1e-4):
    '''Implementation for Taylor Series'''
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2*i + 1)
        t = c * x ** (2*i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


if __name__ == '__main__':
    from dezero import Variable

    x = Variable(np.array(np.pi/4), name='x')
    y = sin(x)
    y.backward()

    print(y.data)
    print(x.grad)

    x.clear_grad(); y.clear_grad()
    y = my_sin(x, threshold=1e-10)
    y.backward()

    y.name = 'y'
    print(y.data)
    print(x.data)

    plot_dot_graph(y, verbose=False, to_file='step27_my_sin_1e10.png')
