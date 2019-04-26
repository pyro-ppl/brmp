import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pyro.contrib.brm import brm, makecode
from pyro.contrib.brm.formula import Formula, Group, _1
from pyro.contrib.brm.design import makedata
from pyro.contrib.brm.utils import print_marginals

def f(x):
    eps = np.random.normal(0, 0.5, x.shape)
    return (4 * x) - 2 + eps

xs = np.random.uniform(0, 5, [10])
ys = f(xs)

# plt.scatter(xs, ys, marker='x')
# plt.show()

df = pd.DataFrame(dict(x=xs, y=ys))

# y ~ 1 + x
formula = Formula('y', [_1, 'x'], [])

# data = makedata(formula, df)
# for (k, v) in data.items():
#     print('========================================')
#     print(k)
#     print('----------------------------------------')
#     print(v)

#code = makecode(formula, df)
#print(code)

print_marginals(brm(formula, df))
