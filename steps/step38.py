import numpy as np
from dezero import Variable
import dezero.functions as F


def test_reshape_func():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.reshape(x, (6,))
    y.backward(retain_grad=True)

    print(x.data)
    print(y)
    print(y.grad)
    print(x.grad)


def test_reshape_in_var():
    x = Variable(np.random.rand(1, 2, 3))
    y = x.reshape((2, 3))
    z = x.reshape([2, 3])
    print(y)
    print(z)


def test_transpose_func():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)
    y.backward(retain_grad=True)
    print(f'x.shape: {x.shape} -> y.shape: {y.shape}')
    assert x.shape == x.grad.shape, 'not match'
    print(f'x.grad.shape: {x.grad.shape}\nx.grad: {x.grad}')


def test_transpose_in_var():
    x = Variable(np.random.rand(2, 3))
    y = x.transpose()
    print(f'y.shape: {y.shape}')
    y = x.T
    print(f'y.T.shape: {y.T.shape}')

    A, B, C, D = 1, 2, 3, 4
    a = np.random.rand(A, B, C, D)
    b = a.transpose(1, 0, 3, 2)
    print(a)
    print(b)


if __name__ == '__main__':
    test_transpose_in_var()