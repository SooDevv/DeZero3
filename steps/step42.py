import numpy as np
import matplotlib.pyplot as plt

from dezero.core import Variable
import dezero.functions as F


# toy dataset
np.random.seed(2021)
x = np.random.rand(100, 1)
y = 5 + 2*x + np.random.rand(100, 1)
plt.scatter(x, y)
plt.show()

x, y = Variable(x), Variable(y)
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


lr = 1e-1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = F.mse(y, y_pred)

    W.clear_grad()
    b.clear_grad()
    loss.backward()

    W.data = -lr * W.grad.data
    b.data = -lr * b.grad.data
    print(f'iter{i}: W {W}, b {b} loss: {loss}')




