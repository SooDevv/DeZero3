import numpy as np

from dezero import Variable
import dezero.functions as F
import dezero.layers as L
from dezero import Layer

np.random.seed(2021)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

model = Layer()
model.linear1 = L.Linear(10)
model.linear2 = L.Linear(1)


def predict(model, x):
    y = model.linear1(x)
    y = F.sigmoid(y)
    y = model.linear2(y)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(model, x)
    loss = F.mse(y, y_pred)

    # linear1.cleargrads()  # step44.py
    # linear2.cleargrads()  # step44.py
    model.cleargrads()
    loss.backward()

    # for l in [linear1, linear2]:  # step44.py
    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(f'step {i}: loss: {loss}')

