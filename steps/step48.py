import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

########################################
# Set hyper parameter
########################################
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 10

########################################
# Load dataset / model / optimizer
########################################
data, target = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

########################################
# Training
########################################
data_size = len(data)
max_iter = math.ceil(data_size / batch_size)
print('max_iter', max_iter)
for epoch in range(max_epoch):
    # shuffle index of data
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # generate mini-batch
        batch_index = index[i * batch_size: (i+1) * batch_size]
        batch_x = data[batch_index]
        batch_t = target[batch_index]

        # update weight
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print(f'Epoch {epoch+1}, loss = {avg_loss:.2f}')
