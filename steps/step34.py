import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F


def test_sin():
    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)

    for i in range(3):
        gx = x.grad
        x.clear_grad()  # x.grad=None
        gx.backward(create_graph=True)
        print(x.grad)


def test2_sin():
    x = Variable(np.linspace(-7, 7, 200))
    y = F.sin(x)
    y.backward(create_graph=True)

    logs = [y.data]

    for i in range(3):
        logs.append(x.grad.data)
        gx = x.grad
        x.clear_grad()
        gx.backward(create_graph=True)

    # 그래프 그리기
    labels = ["y=sin(x)", "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        plt.plot(x.data, logs[i], label=labels[i])
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    test2_sin()