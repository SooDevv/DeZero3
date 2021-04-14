import numpy as np
from dezero import test_mode
import dezero.functions as F


x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print('test mode: ', y)