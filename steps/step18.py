import numpy as np
import weakref
from memory_profiler import profile
import contextlib


class Variable:
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

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        history_set = set()

        def add_func(f):
            if f not in history_set:
                funcs.append(f)
                history_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx if x.grad is None else x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Config:
    enable_backprop = True


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # 아래의 기능들은 역전파시에만 사용
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

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


def square(x):
    return Square()(x)


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


@profile()
def my_func():
    Config.enable_backprop = True
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    y.backward()


@profile()
def my_func2():
    Config.enable_backprop = False
    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)  # preprocessing
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)  # postprocessing


def no_grad():
    return using_config('enable_backprop', False)


if __name__ == '__main__':
    # my_func()
    # my_func2()
    with no_grad():
        x = Variable(np.array(2))
        y = square(x)