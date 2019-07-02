import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pyro.contrib.brm import brm
from pyro.contrib.brm.fit import marginals

def f(x):
    eps = np.random.normal(0, 0.5, x.shape)
    return (4 * x) - 2 + eps

xs = np.random.uniform(0, 5, [10])
ys = f(xs)
df = pd.DataFrame(dict(x=xs, y=ys))

# plt.scatter(xs, ys, marker='x')
# plt.show()

fit = brm('y ~ 1 + x', df)

print(marginals(fit))
#                  mean    sd
# b_intercept     -0.94  0.56
# b_x              3.60  0.25
# sigma            0.93  0.26

print(fit.model.code)
# def model(X, y_obs=None):
#     assert type(X) == torch.Tensor
#     N = X.shape[0]
#     M = 2
#     assert X.shape == (N, M)
#     b_0 = pyro.sample("b_0", dist.Cauchy(torch.tensor(0.0).expand([2]), torch.tensor(1.0).expand([2])).to_event(1))
#     b = torch.cat([b_0])
#     assert b.shape == (M,)
#     mu = torch.mv(X, b)
#     sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
#     with pyro.plate("obs", N):
#         y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)).to_event(0), obs=y_obs)
#     return {'b': b}

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
