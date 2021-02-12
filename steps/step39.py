import numpy as np
from dezero.core import Variable
from dezero import functions as F


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x)
y.backward()
print(y)
print(y.grad)
print(x.grad)

x2 = Variable(np.random.rand(2, 3, 4, 5))
y2 = x2.sum(keepdims=True)
print(y2.shape)
print(y2)