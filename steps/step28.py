import numpy as np
from dezero import Variable


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (1 - x0) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 3e-4
iters = 100000

for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)

    x0.clear_grad(); x1.clear_grad()

    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
    