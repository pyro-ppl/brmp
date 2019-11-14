# Bayesian Regression Models

This is an attempt to implement
a [brms](https://github.com/paul-buerkner/brms)-like library in
Python.

It allows Bayesian regression models to be specified using (a subset
of) the lme4 syntax. Given such a description and a pandas data frame,
the library generates model code and design matrices, targeting
either [Pyro](https://pyro.ai/)
or [NumPyro](https://github.com/pyro-ppl/numpyro).

## Resources

* [API documentation](https://brmp.readthedocs.io/en/latest/).
* [Examples](https://nbviewer.jupyter.org/github/pyro-ppl/brmp/tree/master/brmp/examples/).
  Notebooks showing the library been used to fit models to data.

## Current Status

### Model Specification

#### Formula

Here are some example formulae that the system can handle:

| Formula                                      | Description |
|----|----|
| `y ~ x`                                      | Population-level effects |
| `y ~ 1 + x`                                  ||
| `y ~ x1:x2`                                  | Interaction between variables |
| `y ~ 1 + x0 + (x1 \| z)`                     | Group-level effects |
| `y ~ 1 + x0 + (1 + x1 \| z)`                 ||
| `y ~ 1 + x0 + (1 + x1 \|\| z)`               | No correlation between group coefficients |
| `y ~ 1 + x0 + (1 + x1 \| z1:z2)`             | Grouping by multiple factors (untested) |
| `y ~ 1 + x0 + (x1 \| z0) + (1 + x2 \|\| z1)` | Combinations of the above |


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
sensible way.

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
* `fitted(...)`: This implements some of the functionality available
  in brms through
  the [`fitted`](https://rdrr.io/cran/brms/man/fitted.brmsfit.html)
  and [`predict`](https://rdrr.io/cran/brms/man/predict.brmsfit.html)
  methods.

## Limitations

* All formula terms must be column names. Expressions such as
  `sin(x1)` or `I(x1*x2)` are not supported.
* The `*` operator is not supported. (Though the model `y ~ 1 + x1*x2`
  can be specified with the formula `y ~ 1 + x1 + x2 + x1:x2`.)
* The `/` operator is not supported. (Though the model `y ~ ... |
  g1/g2` can be specified with the formula `y ~ (... | g1) + (... |
  g1:g2)`.)
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
