import numpy as np

from dezero import Variable
import dezero.functions as F
import dezero.layers as L


np.random.seed(2021)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

linear1 = L.Linear(10)
linear2 = L.Linear(1)


def predict(x):
    y = linear1(x)
    y = F.sigmoid(y)
    y = linear2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mse(y, y_pred)

    linear1.cleargrads()
    linear2.cleargrads()
    loss.backward()

    for l in [linear1, linear2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(f'step {i}: loss: {loss}')