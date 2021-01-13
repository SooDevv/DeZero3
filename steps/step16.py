import numpy as np


class Variable:
    '''
        Variable는 변수를 담는 상자, 함수와의 관계 명시, 중요변수에서 backward(역전파) 실행하는 역할
    '''
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx if x.grad is None else x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)

    def claer_grad(self):
        self.grad = None


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    '''
        Function은 함수들의 공통 요소, 변수와의 관계 명시, forward(순전파) 실행
    '''
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 순전파 후 나온 결과 데이터
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]  # 데이터를 변수에 담음.

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *x):
        raise NotImplementedError

    def backward(self, *gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return np.square(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


def square(x):
    return Square()(x)


if __name__ == '__main__':
    x = Variable(np.array(2))
    a = square(x)
    y = add(square(a), square(a))  # y = ((x^2)^2) + ((x^2)^2)
    y.backward()
    print(y.data)
    print(x.grad)