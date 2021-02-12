import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.random.rand(2, 3), name='x')
W = Variable(np.random.rand(3, 4), name='W')
y = F.matmul(x, W)
y.backward()

y.name = 'y'

print(y.shape)
for i in [x, W]:
    print(f'{i.name} shape: {i.grad.shape}')
