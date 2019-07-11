# Bayesian Regression Models

This is an attempt to implement
a [brms](https://github.com/paul-buerkner/brms)-like library in
Python.

It allows Bayesian regression models to be specified using (a subset
of) the lme4 syntax. Given such a description and a pandas data frame,
the library generates model code and design matrices, targeting
either [Pyro](https://pyro.ai/)
or [NumPyro](https://github.com/pyro-ppl/numpyro).

## Current Status

### Model Specification

#### Formula

Here are some example formulae that the system can handle:

| Formula                                      | Description |
|----|----|
| `y ~ x`                                      | Population-level effects |
| `y ~ 1 + x`                                  ||
| `y ~ x1:x2`                                  | Interaction between categorical variables |
| `y ~ 1 + x0 + (x1 \| z)`                     | Group-level effects |
| `y ~ 1 + x0 + (1 + x1 \| z)`                 ||
| `y ~ 1 + x0 + (1 + x1 \|\| z)`               | No correlation between group coefficients |
| `y ~ 1 + x0 + (x1 \| z0) + (1 + x2 \|\| z1)` | Combinations of the above |


The file [`ex1.py`](./ex1.py) shows the model code and data generated
for a number of similar formulae.

#### Priors

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

#### Response Families

The library supports models with either (uni-variate) Gaussian or
Binomial (inc. Bernoulli) distributed responses.

### Inference

The Pyro back end supports both NUTS and SVI for inference. The
NumPyro backend supports only NUTS.

The library includes the following functions for working with
posteriors:

* `marginals(...)`: This produces a model summary similar to that
  obtained by doing `fit <- brm(...) ; fit$fit` in brms.
* `fitted(...)`: This is analogous to
  the [`fitted`](https://rdrr.io/cran/brms/man/fitted.brmsfit.html)
  method in brms.

See the baseball notebook (linked below) for example usage.

### Examples

These
[notebooks](https://nbviewer.jupyter.org/github/null-a/pyro/tree/brmp/pyro/contrib/brm/examples/) show
examples of using the system to fit some simple models.

## Limitations

* Interactions between numeric variables are not supported.
* All formula terms must be column names. Expressions such as
  `sin(x1)` or `I(x1*x2)` are not supported.
* The group syntax `g1:g2` and `g1/g2` is not supported.
* The `*` operator is not supported. (Though the model `y ~ 1 + x1*x2`
  can be specified with the formula `y ~ 1 + x1 + x2 + x1:x2`.)
* The syntax for removing columns is not supported. e.g. `y ~ x - 1`
* The response is always uni-variate.
* Parameters of the response distribution cannot take their values
  from the data. e.g. The number of trials parameter of Binomial can
  only be set to a constant, and cannot vary across rows of the data.
* Only a limited number of response families are supported. In
  particular, Categorical responses (beyond the binary case) are not
  supported.
* Some priors used in the generated code don't match those generated
  by brms. e.g. There's no Half Student-t distribution, setting prior
  parameters based on the data isn't supported.
* The centering data transform, performed by brms to improve sampling
  efficiency, is not implemented.
* This doesn't include any of the fancy stuff brms does, such as its
  extensions to the lme4 grouping syntax, splines, monotonic effects,
  GP terms, etc.
* The `fitted` function does not implement all of the functionality of
  its analogue in brms.
* There are no tools to help with MCMC diagnostics, posterior checks,
  hypothesis testing, etc.
* Lots more, probably...
