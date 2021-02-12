import numpy as np
import matplotlib.pyplot as plt

from dezero import Variable
import dezero.functions as F


# dataset
np.random.seed(2021)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
plt.scatter(x, y)
plt.show()

# initialize weight
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.rand(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.rand(H, O))
b2 = Variable(np.zeros(O))


def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000

# training
for i in range(iters):
    y_pred = predict(x)
    loss = F.mse(y, y_pred)

    W1.clear_grad()
    b1.clear_grad()
    W2.clear_grad()
    b2.clear_grad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:
        print(f'loss: {loss}')
