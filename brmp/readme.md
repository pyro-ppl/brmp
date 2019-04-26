# Bayesian Regression Models

## Formulae

The basic lme4 syntax looks like:

`response ~ pterms + (gterms | group)`

A subset of this can be described using `Formula` and `Group`, for
example:

```python
# y ~ x
Formula('y', [_1, 'x'], [])

# y ~ x1 + x2 + (1 | x3) + (x4 + x5 || x6)
Formula('y',
        [_1, 'x1', 'x2'],
        [Group([_1], 'x3', True), Group([_1, 'x4', 'x5'], 'x6', False)]
```

Functioning model code and design matrices (from a pandas dataframe)
can be generated for this subset. See `ex0.py` for an example.

## Limitations

* All formula terms must be column names. Expressions such as
  `I(x1*x2)` are not supported.
* The group syntax `g1:g2` and `g1/g2` is not supported.
* Interaction between terms in not supported. e.g. `y ~ x1*x2`
* The response is always scalar.
* The response is Gaussian distributed.
* Some priors used in generated code are not sensible.
* Priors cannot be customised.
* Lots more...
