formulae
--------

the basic lme4 syntax looks like: response ~ pterms + (gterms | group)

a subset of this can be described using Formula and Group.

examples:

# y ~ x
Formula('y', ['x'], [])

# y ~ x1 + x2 + (1 | x3) + (x4 + x5 | x6)
Formula('y',
        ['x1', 'x2'],
        [Group([], 'x3'), Group(['x4', 'x5'], 'x6')]

current limitations include:

- all formula terms are column names. expressions are not supported.
  (e.g. I(X1*X2).)
- the group syntax g1:g2 and g1/g2 is not supported.
- interaction between terms in not supported. e.g. y ~ x1 * x2
- the syntax to indicate the correlations between coefficients within
  a group should not be modelled is not supported. e.g. (gterms ||
  group).
- the default intercept term cannot be suppressed.

metadata
--------

the generated model depends on properties of the data set. rather than
require a concrete data set be give, we instead ask for just the
relevant metadata, which is currently just a list of those columns
which are factors (in the R sense). all other columns are assumed to
be numeric.

examples:

- []                    # all cols are numeric
- [Factor("gender", 2)] # gender is a factor with two levels.

other assumptions
-----------------

- the response is a scalar
- the response is Gaussian distributed
