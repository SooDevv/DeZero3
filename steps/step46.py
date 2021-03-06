import numpy as np
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


# prepare dataset
np.random.seed(2021)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# parameter
lr = 0.2
max_iter = 10000
hidden_size = 10

# set model & optimizier
model = MLP((hidden_size, 1), activation=F.sigmoid)
optimizer = optimizers.SGD(lr).setup(model)

# training
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mse(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print(f'{i} iter: {loss}')