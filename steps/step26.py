import numpy as np

from dezero import Variable
from dezero.utils import _dot_var, _dot_func, plot_dot_graph


a = Variable(np.random.rand(2, 3))
a.name = 'a'

print(_dot_var(a))
print(_dot_var(a, verbose=True))

b = Variable(np.array(1.0))
c = Variable(np.array(1.0))
d = b + c
print(_dot_func(d.creator))


def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z


x = Variable(np.array(1.0), name='x')
y = Variable(np.array(1.0), name='y')
z = goldstein(x, y)
z.backward()


z.name = 'z'
plot_dot_graph(z, verbose=False, to_file='step26_goldstein.png')

