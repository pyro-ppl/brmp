"""

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

"""

from collections import namedtuple

# TODO: Add a parser.
# TODO: Make into classes. Add validation. Add repr.
Formula = namedtuple('Formula',
                     ['response',   # response column name
                      'pterms',     # list of population level columns
                      'groups'])    # list of groups
Group = namedtuple('Group',
                   ['gterms',       # list of group level columns
                    'column'])      # name of grouping column
Factor = namedtuple('Factor',
                    ['name',        # column name
                     'num_levels']) # number of levels

def make_metadata_lookup(metadata):
    assert type(metadata) == list
    assert all(type(factor) == Factor for factor in metadata)
    # Turn a list of factors into a dictionary keyed by column name.
    return dict((factor.name, factor) for factor in metadata)

# Computes the number of entries a column adds to the design matrix.
# Each numeric column contributes 1 entry. Each factor contributes
# num_levels-1.

# The latter is because a factor with 4 levels (for example) is coded
# like so in the design matrix:

# x = factor(c(0, 1, 2, 3))
#
# Intercept x1 x2 x3
# 1          0  0  0
# 1          1  0  0
# 1          0  1  0
# 1          0  0  1

# This is always the case when an intercept is present, otherwise
# things are a little more subtle. Without an intercept, the factor
# above would be coded like so if it appears as the first term in e.g.
# pterms.

# x0 x1 x2 x3
#  1  0  0  0
#  0  1  0  0
#  0  0  1  0
#  0  0  0  1

# Subsequent factors are then coded as they would be were an intercept
# present.

def width(col, metadata_lookup):
    if col in metadata_lookup:
        return metadata_lookup[col].num_levels - 1
    else:
        return 1

# This assumes that all dims are event dims.
def std_cauchy(shape):
    assert type(shape) == list
    return 'Cauchy(torch.zeros({}), torch.ones({})).to_event({})'.format(shape, shape, len(shape))

# This assumes that all dims are event dims.
def half_cauchy(scale, shape):
    assert type(scale) == float
    assert type(shape) == list
    return 'HalfCauchy(torch.tensor({}).expand({})).to_event({})'.format(scale, shape, len(shape))

def lkj_corr_cholesky(size, shape):
    assert type(size) == int # the size of the matrix
    assert type(shape) == float # shape parameter of distribution
    return 'LKJCorrCholesky({}, torch.tensor({}))'.format(size, shape)

def std_normal(shape):
    assert type(shape) == list
    return 'Normal(torch.zeros({}), torch.ones({})).to_event({})'.format(shape, shape, len(shape))

def sample(name, distribution):
    return '{} = pyro.sample("{}", dist.{})'.format(name, name, distribution)

def indent(line):
    return '    {}'.format(line)

def method(name, body):
    assert type(body) == list
    return ['def {}():'.format(name)] + [indent(line) for line in body]

def genpopulation():
    pass

# Generates model code for a single group. More specifically, this
# generates code to sample group level priors and to accumulate the
# groups contribution to mu.
def gengroup(i, group, metadata):
    assert type(i) == int # A unique int assigned to each group.
    assert type(group) == Group
    metadata_lookup = make_metadata_lookup(metadata)
    # The column on which we group must be a factor.
    assert group.column in metadata_lookup, 'group column must be a factor'
    groupfactor = metadata_lookup[group.column]

    code = []
    # The number of coefficients per group.
    M = 1 + sum(width(col, metadata_lookup) for col in group.gterms)

    # TODO: Fix the code generated when M=1.
    # The problem is that the LKJ prior is only defined when M > 1.
    # (i.e. there are at least 2 coefficients between which to model
    # correlations.) An easy fix is to wait until we support groups
    # that don't model correlations, and switch to using that when
    # correlations are requested by the spec. but only a single
    # coefficient is present.
    assert M > 1, 'each group must include at least one term in addition to the intercept.'

    # The number of groups.
    N = groupfactor.num_levels

    # This follows the names used in brms.
    # TODO: Use these in the code below for readability/modifiability
    # of generated code.
    code.append('M_{} = {}'.format(i, M))
    code.append('N_{} = {}'.format(i, N))

    # Dummy group-level design matrix and group look-up. (Hard-coded
    # number of data points.)
    for j in range(1, M+1):
        code.append('Z_{}_{} = torch.rand(5)'.format(i, j))
    code.append('J_{} = torch.randint(0, {}, size=[5])'.format(i, N))

    # Model correlations between the coefficients.

    # Prior over correlations.
    # brms uses a shape of 1.0 by default.
    code.append(sample('L_{}'.format(i), lkj_corr_cholesky(M, shape=1.)))

    # Prior over coefficient scales.
    code.append(sample('sd_{}'.format(i), half_cauchy(scale=3.0, shape=[M])))

    # Prior over a matrix of unscaled/uncorrelated coefficients.
    code.append(sample('z_{}'.format(i), std_normal([M, N])))

    # Compute the final (scaled, correlated) coefficients.
    code.append('r_{} = torch.mm(torch.mm(torch.diag(sd_{}), L_{}), z_{}).transpose(0, 1)'.format(i, i, i, i))

    # The following has a similar structure to the code generated by
    # brms, though it's not clear that this is optimal for PyTorch.
    # One alternative is to have a single design matrix for the group,
    # e.g. Z_1, and to do the following (rather than looping over each
    # coefficient): mu = mu + torch.sum(Z_1 * r_1[J_1], 1)

    # This is vectorised over N and M, but allocates a large
    # intermediate tensor. (Though I don't think this is worse than
    # the current implementation.) Perhaps this can be avoided with
    # einsum notation.

    for j in range(M):
        code.append('r_{}_{} = r_{}[:, {}]'.format(i, j+1, i, j))
    for j in range(M):
        code.append('mu = mu + r_{}_{}[J_{}] * Z_{}_{}'.format(i, j+1, i, i, j+1))

    return code

def genresponse():
    pass

def genmodel(formula, metadata):
    assert type(formula) == Formula
    metadata_lookup = make_metadata_lookup(metadata)

    # Since we're not working with any data we don't know how many
    # rows the design matrix will have, so we hard code for now.
    N = 5
    # The number of columns of the design matrix. We assume the
    # presence of an intercept.
    M = 1 + sum(width(col, metadata_lookup) for col in formula.pterms)

    body = []

    # Dummy design matrix.
    body.append('N = {}'.format(N))
    body.append('M = {}'.format(M))
    body.append('X = torch.rand(N, M)')

    # Population level
    # --------------------------------------------------

    # Prior over b. (The population level coefficients.)
    # TODO: brms uses an improper uniform here.
    body.append(sample('b', std_cauchy(shape=[M])))
    # Compute mu.
    body.append('mu = torch.mv(X, b)')

    # Group level
    # --------------------------------------------------
    for i, group in enumerate(formula.groups):
        # Use 1 indexed groups to ease comparison of generate code
        # with code generated by brms.
        body.extend(gengroup(i+1, group, metadata))

    # Response
    # --------------------------------------------------

    # Prior over the std. dev. of the response distribution.
    # TODO: brms uses a Half Student-t here, with a scale computed
    # from the data.
    body.append(sample('sigma', half_cauchy(scale=3., shape=[1])))

    # TODO: Add the observation.
    # TODO: Add plate?
    body.append(sample('y', 'Normal(mu, sigma.expand(N)).to_event(1)'))

    body.append('return dict(b=b, sigma=sigma, y=y)')

    return '\n'.join(method('model', body))

def eval_model(model_code):
    import torch
    import pyro
    import pyro.distributions as dist
    g = locals()
    exec(model_code, g)
    return g['model']

def main():
    formula = Formula('y', [], [Group(['x2'], 'x1'), Group(['x3'], 'x2')])
    code = genmodel(formula, metadata=[Factor('x1', 3), Factor('x2', 4)])
    print(code)
    model = eval_model(code)
    print(model())

if __name__ == '__main__':
    main()
