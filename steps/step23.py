if '__file__' in globals():
    import os, sys
    parent_path = os.path.join(os.path.dirname(__file__), '..')  # /Users/user/Documents/Git/DeZero3/steps/..
    sys.path.append(parent_path)


import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)