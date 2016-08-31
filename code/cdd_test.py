import numpy as np
import itertools

data = np.random.randn(100, 10, 10)
boo = data < 0.254
boo = boo.reshape(100, 100)

def runs_of_ones(bits):
    return [sum(g) for b, g in itertools.groupby(bits) if b]

res = []
for i in range(boo.shape[1]):
    mx = max(runs_of_ones(boo[:, i]))
    res.append(mx)


