import numpy as np

from dezero import Variable
from dezero.utils import _dot_var, _dot_func


x = Variable(np.random.rand(2, 3))
x.name = 'x'

print(_dot_var(x))
print(_dot_var(x, verbose=True))

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1
print(_dot_func(y.creator))