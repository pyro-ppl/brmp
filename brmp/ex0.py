import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pyro.contrib.brm import brm, makecode
from pyro.contrib.brm.formula import Formula, Group, _1
from pyro.contrib.brm.design import makedata
from pyro.contrib.brm.fit import print_marginals

def f(x):
    eps = np.random.normal(0, 0.5, x.shape)
    return (4 * x) - 2 + eps

xs = np.random.uniform(0, 5, [10])
ys = f(xs)
df = pd.DataFrame(dict(x=xs, y=ys))

# plt.scatter(xs, ys, marker='x')
# plt.show()

# y ~ 1 + x
formula = Formula('y', [_1, 'x'], [])
fit = brm(formula, df)

print_marginals(fit)
# ==================================================
# b
# -- mean ------------------------------------------
# tensor([-1.9719,  3.9674])
# -- stddev ----------------------------------------
# tensor([0.3830, 0.1312])
# ==================================================
# sigma
# -- mean ------------------------------------------
# tensor([0.4817])
# -- stddev ----------------------------------------
# tensor([0.1592])

print(fit.code)
# def model(X, y_obs=None):
#     assert type(X) == torch.Tensor
#     N = X.shape[0]
#     M = 2
#     assert X.shape == (N, M)
#     b = pyro.sample("b", dist.Cauchy(torch.zeros([2]), torch.ones([2])).to_event(1))
#     mu = torch.mv(X, b)
#     sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
#     with pyro.plate("obs", N):
#         y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)), obs=y_obs)
#     return dict(b=b, sigma=sigma, y=y)

print(fit.data)
# {'X': tensor([[1.0000, 0.6350],
#         [1.0000, 4.6728],
#         [1.0000, 3.8269],
#         [1.0000, 2.1532],
#         [1.0000, 2.9952],
#         [1.0000, 2.2982],
#         [1.0000, 0.9645],
#         [1.0000, 1.6291],
#         [1.0000, 1.7237],
#         [1.0000, 4.2023]]), 'y_obs': tensor([ 0.8886, 16.3859, 13.2107,  6.7207,  9.9708,  6.4037,  1.5546,  4.0902,
#          5.1075, 15.1358])}
