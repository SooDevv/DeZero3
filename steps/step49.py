import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


max_epcoh = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral()
model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size/batch_size)

for epoch in range(max_epcoh):
    idx = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_idx = idx[i * batch_size: (i+1) * batch_size]
        batch = [train_set[i] for i in batch_idx]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print(f'{epoch+1}, loss {avg_loss:.2f}')
