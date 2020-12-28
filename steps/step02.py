class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2


if __name__ == '__main__':
    import numpy as np
    x = Variable(np.array(10))
    f = Square()  # __call__에 의해 함수를 인스턴스화 하고 메소드 호츨 가능.
    y = f(x)
    print(type(y))  # <class '__main__.Variable'>
    print(y.data)  # 100
