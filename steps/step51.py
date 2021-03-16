import matplotlib.pyplot as plt

import dezero
import dezero.functions as F
from dezero import Dataloader, optimizers
from dezero.models import MLP

max_epoch = 100
batch_size = 100
hidden_size = 1000

tr_dataset = dezero.datasets.MNIST(train=True)
test_dataset = dezero.datasets.MNIST(train=False)
tr_loader = Dataloader(tr_dataset, batch_size)
test_loader = Dataloader(test_dataset, batch_size, shuffle=False)

# model = MLP((hidden_size, 10)) + sgd -> test_acc = 92.78%
model = MLP((hidden_size, 300, 10), activation=F.relu)  # +adam -> test_acc = 97.74%
optimizer = optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in tr_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print(f'{epoch + 1}')
    print(f'train loss: {(sum_loss / len(tr_dataset)):.4f}, \
        acc: {(sum_acc / len(tr_dataset)):.4f}')

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(f'test loss: {(sum_loss / len(test_dataset)):.4f} \
        test acc: {(sum_acc / len(test_dataset)):.4f}')