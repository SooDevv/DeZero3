import numpy as np
import dezero
from dezero import Model
from dezero import SeqDataLoader
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt


def simple_test():
    train_set = dezero.datasets.SinCurve(train=True)
    dataloader = SeqDataLoader(train_set, batch_size=3)
    x, t = next(dataloader)
    print(x)
    print('--------------')
    print(t)


def train_lstm():
    max_epoch = 100
    batch_size = 30
    hidden_size = 100
    bptt_length = 30

    train_set = dezero.datasets.SinCurve(train=True)
    dataloader = SeqDataLoader(train_set, batch_size=batch_size)
    seqlen = len(train_set)

    class BetterRNN(Model):
        def __init__(self, hidden_size, out_size):
            super().__init__()
            self.rnn = L.LSTM(hidden_size)
            self.fc = L.Linear(out_size)

        def reset_state(self):
            self.rnn.reset_state()

        def forward(self, x):
            y = self.rnn(x)
            y = self.fc(x)
            return y

    model = BetterRNN(hidden_size, 1)
    optimizer = dezero.optimizers.Adam().setup(model)

    for epoch in range(max_epoch):
        model.reset_state()
        loss, count = 0, 0

        for x, t in dataloader:
            y = model(x)
            loss += F.mse(y, t)
            count += 1
            if count % bptt_length == 0 or count == seqlen:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
        avg_loss= float(loss.data)
        print(f'|epoch {epoch+1} | loss {avg_loss}')

    return model


def show_predict(model):
    xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
    model.reset_state()
    pred_list = []

    with dezero.no_grad():
        for x in xs:
            x = np.array(x).reshape(1, 1)
            y = model(x)
            pred_list.append(float(y.data))

    plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
    plt.plot(np.arange(len(xs)), pred_list, label='predict')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model = train_lstm()
    show_predict(model)