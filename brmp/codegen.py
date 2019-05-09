from .formula import Formula, Group
from .design import width
from .priors import Prior

def gendist(prior, shape):
    assert type(prior) == Prior
    assert type(shape) == list
    # This is likely only useful when `len(shape) == 1`. And even then
    # it's not very flexible since we assume parameters are given as
    # scalars which we can always expand into parameter vectors.
    assert len(shape) == 1 # See comment above.
    # TODO: Would there be any perf. increase associated with using
    # e.g. `torch.zeros(shape)` over `torch.tensor(0.).expand(shape)`?
    params_code = ['torch.tensor({}).expand({})'.format(param, shape) for param in prior.parameters]
    return '{}({}).to_event({})'.format(prior.family, ', '.join(params_code), len(shape))

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

def sample(name, distribution, obs=None):
    args = ['"{}"'.format(name),
            'dist.{}'.format(distribution)]
    if obs:
        args.append('obs={}'.format(obs))
    return '{} = pyro.sample({})'.format(name, ', '.join(args))

def indent(line):
    return '    {}'.format(line)

def method(name, parameters, body):
    assert type(body) == list
    assert type(parameters) == list
    return ['def {}({}):'.format(name, ', '.join(parameters))] + [indent(line) for line in body]

def comment(s):
    return '# {}'.format(s)

# Generates model code for a single group. More specifically, this
# generates code to sample group level priors and to accumulate the
# groups contribution to mu.
def gengroup(i, group, metadata):
    assert type(i) == int # A unique int assigned to each group.
    assert type(group) == Group
    assert type(metadata) == dict
    # The column on which we group must be a factor.
    assert group.column in metadata, 'group column must be a factor'
    groupfactor = metadata[group.column]

    code = ['']
    code.append(comment('[{}] {}'.format(i, group)))

    # The number of coefficients per level.
    M_i = width(group.gterms, metadata)

    # The number of levels.
    N_i = len(groupfactor.levels)

    # This follows the names used in brms.
    code.append('M_{} = {} # Number of coeffs'.format(i, M_i))
    code.append('N_{} = {} # Number of levels'.format(i, N_i))

    code.append('assert type(Z_{}) == torch.Tensor'.format(i))
    code.append('assert Z_{}.shape == (N, M_{}) # N x {}'.format(i, i, M_i))
    code.append('assert type(J_{}) == torch.Tensor'.format(i))
    code.append('assert J_{}.shape == (N,)'.format(i))

    # Prior over coefficient scales.
    code.append(sample('sd_{}'.format(i), half_cauchy(scale=3.0, shape=[M_i])))
    code.append('assert sd_{}.shape == (M_{},) # {}'.format(i, i, M_i))

    # Prior over a matrix of unscaled/uncorrelated coefficients. This
    # is similar to the brms generated Stan code. An alternative would
    # be to pass `torch.mm(torch.diag(sd_{}), L_{})` as the
    # `scale_tril` argument of a `MultivariateNormal`. Is there any
    # significant different between these two approaches?
    code.append(sample('z_{}'.format(i), std_normal([M_i, N_i])))
    code.append('assert z_{}.shape == (M_{}, N_{}) # {} x {}'.format(i, i, i, M_i, N_i))

    if group.corr and M_i > 1:
        # Model correlations between the coefficients.

        # Prior over correlations.
        # brms uses a shape of 1.0 by default.
        code.append(sample('L_{}'.format(i), lkj_corr_cholesky(M_i, shape=1.)))
        code.append('assert L_{}.shape == (M_{}, M_{}) # {} x {}'.format(i, i, i, M_i, M_i))

        # Compute the final (scaled, correlated) coefficients.

        # When L_i is the identity matrix (representing no
        # correlation) the following computation of r_i is equivalent
        # to that for r_i in the case where we don't model
        # correlations between coefficients. i.e. the other branch of
        # this conditional.
        code.append('r_{} = torch.mm(torch.mm(torch.diag(sd_{}), L_{}), z_{}).transpose(0, 1)'.format(i, i, i, i))
    else:
        # Compute the final (scaled) coefficients.
        code.append('r_{} = (z_{} * sd_{}.unsqueeze(1)).transpose(0, 1)'.format(i, i, i))

    code.append('assert r_{}.shape == (N_{}, M_{}) # {} x {}'.format(i, i, i, N_i, M_i))

    # The following has a similar structure to the code generated by
    # brms (in order to ease the comparison of generated code), though
    # it's not clear that this will have optimal performance in
    # PyTorch.

    # TODO: One alternative is the following (rather than looping over
    # each coefficient):

    # mu = mu + torch.sum(Z_1 * r_1[J_1], 1)

    # This is vectorised over N and M, but allocates a large
    # intermediate tensor `r_1[J_1]`. (Though I don't think this is
    # worse than the current implementation.) Can this be avoided? (I
    # guess einsum doesn't help because we'd have nested indices?)

    for j in range(M_i):
        code.append('r_{}_{} = r_{}[:, {}]'.format(i, j+1, i, j))
    for j in range(M_i):
        code.append('Z_{}_{} = Z_{}[:, {}]'.format(i, j+1, i, j))
    for j in range(M_i):
        code.append('mu = mu + r_{}_{}[J_{}] * Z_{}_{}'.format(i, j+1, i, i, j+1))

    return code

def genmodel(formula, metadata):
    assert type(formula) == Formula
    assert type(metadata) == dict
    num_groups = len(formula.groups)

    body = []

    body.append('assert type(X) == torch.Tensor')
    body.append('N = X.shape[0]')

    # The number of columns of the design matrix. We assume the
    # presence of an intercept.
    M = width(formula.pterms, metadata)
    body.append('M = {}'.format(M))
    body.append('assert X.shape == (N, M)')

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

    body.append('with pyro.plate("obs", N):')
    body.append(indent(sample('y', 'Normal(mu, sigma.expand(N))', 'y_obs')))

    body.append('return dict(b=b, sigma=sigma, y=y)')

    params = (['X'] +
              ['Z_{}'.format(i+1) for i in range(num_groups)] +
              ['J_{}'.format(i+1) for i in range(num_groups)] +
              ['y_obs=None'])
    return '\n'.join(method('model', params, body))

def eval_model(model_code):
    import torch
    import pyro
    import pyro.distributions as dist
    g = locals()
    exec(model_code, g)
    return g['model']
