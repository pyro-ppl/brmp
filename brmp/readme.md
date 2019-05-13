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
  `I(x1*x2)` are not supported.
* The group syntax `g1:g2` and `g1/g2` is not supported.
* Interaction between terms in not supported. e.g. `y ~ x1*x2`
* The response is always scalar.
* The response is Gaussian distributed.
* Some priors used in generated code are not sensible.
* Not all priors can be customised.
* Lots more...
