# Bayesian Regression Models

## Current Status

The core of this library is the ability to generate model code and
design matrices from a model specified using (a subset of) the lme4
syntax and a pandas dataframe. See [`ex1.py`](./ex1.py) for some
examples.

In addition, a thin wrapper around this functionality aims to
eventually provide a brms-like interface for fitting models.
See [`ex0.py`](./ex0.py) for an example of this.

## Limitations

* All formula terms must be column names. Expressions such as
  `sin(x1)` or `I(x1*x2)` are not supported.
* The group syntax `g1:g2` and `g1/g2` is not supported.
* Interaction between terms in not supported. e.g. `y ~ x1*x2`
* The syntax for removing columns is not supported. e.g. `y ~ x - 1`
* The response is always scalar.
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
* Lots more...
