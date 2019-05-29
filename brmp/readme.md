# Bayesian Regression Models

## Current Status

### Formula

The core of this library is the ability to generate model code and
design matrices from a model specified using (a subset of) the lme4
syntax and a pandas dataframe. Here are some example formulae that the
system can handle:

| Formula                                   | Description |
|----|----|
| `y ~ x`                                   | Population-level effects |
| `y ~ 1 + x`                               ||
| `y ~ 1 + x0 + (x1 \| z)`                   | Group-level effects |
| `y ~ 1 + x0 + (1 + x1 \| z)`               ||
| `y ~ 1 + x0 + (1 + x1 \|\| z)`              | No correlation between group coefficients |
| `y ~ 1 + x0 + (x1 \| z0) + (1 + x2 \|\| z1)` | Multiple groups |


The file [`ex1.py`](./ex1.py) shows the model code and data generated
for a number of similar formulae.

### Priors

Custom priors can be specified at various levels of granularity. For
example, users can specify:

* A prior to be used for every population-level coefficient.
* A prior to be used for a particular population-level coefficient.
  (The system is aware of the coding used for categorical
  columns/factors in the data frame, which allows priors to be
  assigned to the coefficient corresponding to a particular level of a
  factor.)
* A prior to be used for all columns of the standard deviation
  vector in every group.
* A prior to be used for all columns of the standard deviation
  vector in a particular group.
* A prior to be used for a particular coefficient of the standard
  deviation vector in a particular group.
* etc.

Users can give multiple such specifications and they combine in a
sensible way. See [`ex1.py`](./ex1.py#L141) for a simple example of
this.

### Interface

A thin wrapper around the core functionality aims to eventually
provide a brms-like interface for fitting models.
See
[this notebook](https://nbviewer.jupyter.org/github/null-a/pyro/blob/brmp/pyro/contrib/brm/examples/Example1.ipynb) for
an example of using the system to fit a simple model

## Limitations

* All formula terms must be column names. Expressions such as
  `sin(x1)` or `I(x1*x2)` are not supported.
* The group syntax `g1:g2` and `g1/g2` is not supported.
* Interaction between terms in not supported. e.g. `y ~ x1*x2`
* The syntax for removing columns is not supported. e.g. `y ~ x - 1`
* The response is always uni-variate.
* The response is Gaussian or Bernoulli distributed. In particular
  Categorical responses (beyond the binary case) are not supported.
* Some priors used in the generated code don't match those generated
  by brms. e.g. There's no Half Student-t distribution, setting prior
  parameters based on the data isn't supported.
* The centering data transform, performed by brms to improve sampling
  efficiency, is not implemented.
* This doesn't include any of the fancy stuff brms does, such as its
  extensions to the lme4 grouping syntax, splines, monotonic effects,
  GP terms, etc.
* There are no tools to help with MCMC diagnostics, posterior checks,
  hypothesis testing, etc.
* Lots more, probably...
