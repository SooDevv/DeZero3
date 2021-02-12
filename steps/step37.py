import numpy as np
from dezero.core import Variable
from dezero import functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]), 'x')
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]), 'c')
t = x + c
y = F.sum(t)
y.backward(retain_grad=True)

t.name = 't'
y.name = 'y'

print(f't: {t}')
print(f'y: {y}')
for i in [y, t, x, c]:
    assert i.shape == i.grad.shape, 'not match shape'
    print(f'{i.name}.grad : {i.grad}')
