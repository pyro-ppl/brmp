# Examples

## Example 1

`y ~ x`

### Pyro

Spec:

```python
formula = Formula('y', ['x'], [])
metadata = []
code = genmodel(formula, metadata)
```

Generated code:

```python
def model():
    N = 5
    M = 2
    X = torch.rand(N, M)
    b = pyro.sample("b", dist.Cauchy(torch.zeros([2]), torch.ones([2])).to_event(1))
    mu = torch.mv(X, b)
    sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
    y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)).to_event(1))
    return dict(b=b, sigma=sigma, y=y)
```

### Stan via brms

```stan
// make_stancode(bf(y ~ x, center=0), data=df)
// generated with brms 2.8.0
functions {
}
data {
  int<lower=1> N;  // number of observations
  vector[N] Y;  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
}
parameters {
  vector[K] b;  // population-level effects
  real<lower=0> sigma;  // residual SD
}
transformed parameters {
}
model {
  vector[N] mu = X * b;
  // priors including all constants
  target += student_t_lpdf(sigma | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  // likelihood including all constants
  if (!prior_only) {
    target += normal_lpdf(Y | mu, sigma);
  }
}
generated quantities {
}
```

## Example 2

`y ~ x1 + (x2 | x3) + (x4 + x5 | x6)`

### Pyro

Spec:

```python
formula = Formula('y', ['x1'], [Group(['x2'], 'x3'), Group(['x4', 'x5'], 'x6')])
metadata = [Factor('x2', 2), Factor('x3', 3), Factor('x6', 4)]
code = genmodel(formula, metadata)
```

Generated code:

```python
def model():
    N = 5
    M = 2
    X = torch.rand(N, M)
    b = pyro.sample("b", dist.Cauchy(torch.zeros([2]), torch.ones([2])).to_event(1))
    mu = torch.mv(X, b)
    M_1 = 2
    N_1 = 3
    Z_1_1 = torch.rand(5)
    Z_1_2 = torch.rand(5)
    J_1 = torch.randint(0, 3, size=[5])
    L_1 = pyro.sample("L_1", dist.LKJCorrCholesky(2, torch.tensor(1.0)))
    sd_1 = pyro.sample("sd_1", dist.HalfCauchy(torch.tensor(3.0).expand([2])).to_event(1))
    z_1 = pyro.sample("z_1", dist.Normal(torch.zeros([2, 3]), torch.ones([2, 3])).to_event(2))
    r_1 = torch.mm(torch.mm(torch.diag(sd_1), L_1), z_1).transpose(0, 1)
    r_1_1 = r_1[:, 0]
    r_1_2 = r_1[:, 1]
    mu = mu + r_1_1[J_1] * Z_1_1
    mu = mu + r_1_2[J_1] * Z_1_2
    M_2 = 3
    N_2 = 4
    Z_2_1 = torch.rand(5)
    Z_2_2 = torch.rand(5)
    Z_2_3 = torch.rand(5)
    J_2 = torch.randint(0, 4, size=[5])
    L_2 = pyro.sample("L_2", dist.LKJCorrCholesky(3, torch.tensor(1.0)))
    sd_2 = pyro.sample("sd_2", dist.HalfCauchy(torch.tensor(3.0).expand([3])).to_event(1))
    z_2 = pyro.sample("z_2", dist.Normal(torch.zeros([3, 4]), torch.ones([3, 4])).to_event(2))
    r_2 = torch.mm(torch.mm(torch.diag(sd_2), L_2), z_2).transpose(0, 1)
    r_2_1 = r_2[:, 0]
    r_2_2 = r_2[:, 1]
    r_2_3 = r_2[:, 2]
    mu = mu + r_2_1[J_2] * Z_2_1
    mu = mu + r_2_2[J_2] * Z_2_2
    mu = mu + r_2_3[J_2] * Z_2_3
    sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
    y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)).to_event(1))
    return dict(b=b, sigma=sigma, y=y)
```

### Stan via brms

```stan
// str(df)
// 'data.frame':	5 obs. of  7 variables:
// $ y : num  1 1 1 1 1
// $ x1: num  1 1 1 1 1
// $ x2: Factor w/ 2 levels "0","1": 1 1 2 2 2
// $ x3: Factor w/ 3 levels "0","1","2": 1 1 2 2 3
// $ x4: num  1 1 1 1 1
// $ x5: num  1 1 1 1 1
// $ x6: Factor w/ 4 levels "0","1","2","3": 1 2 3 4 4
// make_stancode(bf(y ~ x1 + (x2 | x3) + (x4 + x5 | x6), center=0), data=df)
// generated with brms 2.8.0
functions {
}
data {
  int<lower=1> N;  // number of observations
  vector[N] Y;  // response variable
  int<lower=1> K;  // number of population-level effects
  matrix[N, K] X;  // population-level design matrix
  // data for group-level effects of ID 1
  int<lower=1> N_1;
  int<lower=1> M_1;
  int<lower=1> J_1[N];
  vector[N] Z_1_1;
  vector[N] Z_1_2;
  int<lower=1> NC_1;
  // data for group-level effects of ID 2
  int<lower=1> N_2;
  int<lower=1> M_2;
  int<lower=1> J_2[N];
  vector[N] Z_2_1;
  vector[N] Z_2_2;
  vector[N] Z_2_3;
  int<lower=1> NC_2;
  int prior_only;  // should the likelihood be ignored?
}
transformed data {
}
parameters {
  vector[K] b;  // population-level effects
  real<lower=0> sigma;  // residual SD
  vector<lower=0>[M_1] sd_1;  // group-level standard deviations
  matrix[M_1, N_1] z_1;  // unscaled group-level effects
  // cholesky factor of correlation matrix
  cholesky_factor_corr[M_1] L_1;
  vector<lower=0>[M_2] sd_2;  // group-level standard deviations
  matrix[M_2, N_2] z_2;  // unscaled group-level effects
  // cholesky factor of correlation matrix
  cholesky_factor_corr[M_2] L_2;
}
transformed parameters {
  // group-level effects
  matrix[N_1, M_1] r_1 = (diag_pre_multiply(sd_1, L_1) * z_1)';
  vector[N_1] r_1_1 = r_1[, 1];
  vector[N_1] r_1_2 = r_1[, 2];
  // group-level effects
  matrix[N_2, M_2] r_2 = (diag_pre_multiply(sd_2, L_2) * z_2)';
  vector[N_2] r_2_1 = r_2[, 1];
  vector[N_2] r_2_2 = r_2[, 2];
  vector[N_2] r_2_3 = r_2[, 3];
}
model {
  vector[N] mu = X * b;
  for (n in 1:N) {
    mu[n] += r_1_1[J_1[n]] * Z_1_1[n] + r_1_2[J_1[n]] * Z_1_2[n] + r_2_1[J_2[n]] * Z_2_1[n] + r_2_2[J_2[n]] * Z_2_2[n] + r_2_3[J_2[n]] * Z_2_3[n];
  }
  // priors including all constants
  target += student_t_lpdf(sigma | 3, 0, 10)
    - 1 * student_t_lccdf(0 | 3, 0, 10);
  target += student_t_lpdf(sd_1 | 3, 0, 10)
    - 2 * student_t_lccdf(0 | 3, 0, 10);
  target += normal_lpdf(to_vector(z_1) | 0, 1);
  target += lkj_corr_cholesky_lpdf(L_1 | 1);
  target += student_t_lpdf(sd_2 | 3, 0, 10)
    - 3 * student_t_lccdf(0 | 3, 0, 10);
  target += normal_lpdf(to_vector(z_2) | 0, 1);
  target += lkj_corr_cholesky_lpdf(L_2 | 1);
  // likelihood including all constants
  if (!prior_only) {
    target += normal_lpdf(Y | mu, sigma);
  }
}
generated quantities {
  // omitted ...
}
```
