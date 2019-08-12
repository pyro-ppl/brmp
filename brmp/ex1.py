import numpy as np
import pandas as pd

from pyro.contrib.brm import makecode
from pyro.contrib.brm.formula import parse
from pyro.contrib.brm.design import makedata
from pyro.contrib.brm.priors import Prior
from pyro.contrib.brm.family import Normal, Cauchy

# --------------------------------------------------
# Ex 1. Population-level effects only
# --------------------------------------------------

# Here's a simple formula:
f1 = 'y ~ 1 + x'

# Given a data frame:
df1a = pd.DataFrame(dict(y=[0., 1., 2.],
                         x=[10., 20., 30.]))

# ... we can generate model code and a design matrix:

print(makecode(parse(f1), df1a, Normal, []))
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

print(makedata(parse(f1), df1a))
# {'X': tensor([[ 1., 10.],
#         [ 1., 20.],
#         [ 1., 30.]]), 'y_obs': tensor([0., 1., 2.])}

# Here we use the same formula (`y ~ 1 + x`), but this time we use a
# data frame in which the `x` column is a factor. This factor is
# appropriately coded in the design matrix, and the vector of
# coefficients (`b`) is extended to the appropriate length in the
# generated model code.

df1b = pd.DataFrame(dict(y=[0., 0., 0.],
                         x=pd.Categorical(['a', 'b', 'c'])))

print(makecode(parse(f1), df1b, Normal, []))
# def model(X, y_obs=None):
#     assert type(X) == torch.Tensor
#     N = X.shape[0]
#     M = 3
#     assert X.shape == (N, M)
#     b_0 = pyro.sample("b_0", dist.Cauchy(torch.tensor(0.0).expand([3]), torch.tensor(1.0).expand([3])).to_event(1))
#     b = torch.cat([b_0])
#     assert b.shape == (M,)
#     mu = torch.mv(X, b)
#     sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
#     with pyro.plate("obs", N):
#         y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)).to_event(0), obs=y_obs)
#     return {'b': b}

print(makedata(parse(f1), df1b))
# {'X': tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 0., 1.]]), 'y_obs': tensor([0., 0., 0.])}

# --------------------------------------------------
# Ex 2. Population and group-level effects
# --------------------------------------------------

# Here's a formula with group-level effects:
f2 = 'y ~ x1 + (1 + x2 | x3)'

# ... and a data frame:
df2 = pd.DataFrame(dict(y=[0., 1., 2.],
                        x1=[1., 2., 3.],
                        x2=[10., 20., 30.],
                        x3=pd.Categorical(['a', 'b', 'c'])))

# And the generated model code and design matrices etc.:

print(makecode(parse(f2), df2, Normal, []))
# def model(X, Z_0, J_0, y_obs=None):
#     assert type(X) == torch.Tensor
#     N = X.shape[0]
#     M = 1
#     assert X.shape == (N, M)
#     b_0 = pyro.sample("b_0", dist.Cauchy(torch.tensor(0.0).expand([1]), torch.tensor(1.0).expand([1])).to_event(1))
#     b = torch.cat([b_0])
#     assert b.shape == (M,)
#     mu = torch.mv(X, b)

#     # [0] Factor(name='x3', levels=['a', 'b', 'c'])
#     M_0 = 2 # Number of coeffs
#     N_0 = 3 # Number of levels
#     assert type(Z_0) == torch.Tensor
#     assert Z_0.shape == (N, M_0) # N x 2
#     assert type(J_0) == torch.Tensor
#     assert J_0.shape == (N,)
#     sd_0_0 = pyro.sample("sd_0_0", dist.HalfCauchy(torch.tensor(3.0).expand([2])).to_event(1))
#     sd_0 = torch.cat([sd_0_0])
#     assert sd_0.shape == (M_0,) # 2
#     z_0 = pyro.sample("z_0", dist.Normal(torch.tensor(0.0).expand([2, 3]), torch.tensor(1.0).expand([2, 3])).to_event(2))
#     assert z_0.shape == (M_0, N_0) # 2 x 3
#     L_0 = pyro.sample("L_0", dist.LKJCorrCholesky(2, torch.tensor(1.0)))
#     assert L_0.shape == (M_0, M_0) # 2 x 2
#     r_0 = torch.mm(torch.mm(torch.diag(sd_0), L_0), z_0).transpose(0, 1)
#     assert r_0.shape == (N_0, M_0) # 3 x 2
#     r_0_1 = r_0[:, 0]
#     r_0_2 = r_0[:, 1]
#     Z_0_1 = Z_0[:, 0]
#     Z_0_2 = Z_0[:, 1]
#     mu = mu + r_0_1[J_0] * Z_0_1
#     mu = mu + r_0_2[J_0] * Z_0_2
#     sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
#     with pyro.plate("obs", N):
#         y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)).to_event(0), obs=y_obs)
#     return {'b': b, 'sd_0': sd_0, 'r_0': r_0}




print(makedata(parse(f2), df2))
# {'X': tensor([[1.],
#         [2.],
#         [3.]]), 'y_obs': tensor([0., 1., 2.]), 'Z_0': tensor([[ 1., 10.],
#         [ 1., 20.],
#         [ 1., 30.]]), 'J_0': tensor([0, 1, 2])}



# (Note that modelling of correlations between group-level coefficient
# can be turned off using the `(terms || col)` syntax.)

# --------------------------------------------------
# Ex 3. Custom priors
# --------------------------------------------------

# Here's a simple example in which we specify custom priors for the
# population-level coefficients.

#
f3 = 'y ~ 1 + x1 + x2'
df1a = pd.DataFrame(dict(y=[0., 1., 2.],
                         x1=[1., 2., 3.],
                         x2=[10., 20., 30.]))

# Here we describe the priors:
priors3 = [
    # All population-level coefficients should use a `Normal(0, 10)`
    # prior.
    Prior(('b',), Normal(0., 10.)),
    # The intercept coefficient should use a `Cauchy(0, 100)` prior.
    # This takes precedence over the less specific request above.
    Prior(('b', 'intercept'), Cauchy(0., 100.)),
]

# Here's the code this generates:

print(makecode(parse(f3), df1a, Normal, priors3))
# def model(X, y_obs=None):
#     assert type(X) == torch.Tensor
#     N = X.shape[0]
#     M = 3
#     assert X.shape == (N, M)
#     b_0 = pyro.sample("b_0", dist.Cauchy(torch.tensor(0.0).expand([1]), torch.tensor(100.0).expand([1])).to_event(1))
#     b_1 = pyro.sample("b_1", dist.Normal(torch.tensor(0.0).expand([2]), torch.tensor(10.0).expand([2])).to_event(1))
#     b = torch.cat([b_0, b_1])
#     assert b.shape == (M,)
#     mu = torch.mv(X, b)
#     sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
#     with pyro.plate("obs", N):
#         y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)).to_event(0), obs=y_obs)
#     return {'b': b}
