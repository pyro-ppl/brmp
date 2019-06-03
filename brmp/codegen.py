from .formula import Formula
from .priors import Prior
from .family import Family, LinkFn, getfamily
from .model import Model, Group

def gendist(family, args, shape, batch):
    assert type(family) == Family
    assert type(args) == list
    assert len(args) == len(family.params)
    # Floats are expanded to `shape`, string are assumed to be literal
    # code.
    assert all(type(arg) in [float, str] for arg in args)
    assert type(shape) == list
    # Dimensions are ints (when statically known) or strings (when
    # know at run-time only).
    assert all(type(dim) in [int, str] for dim in shape)
    assert type(batch) == bool
    eventdims = len(shape) - 1 if batch else len(shape)
    args_code = [arg if type(arg) == str else 'torch.tensor({}).expand({})'.format(arg, shape) for arg in args]
    # TODO: When len(shape) == 1 and batch==True, then this
    # `to_event()` call is a no-op, and could therefore be dropped as
    # an optimisation. (Eqv., when eventdims == 0?)
    return '{}({}).to_event({})'.format(family.name, ', '.join(args_code), eventdims)

def lkj_corr_cholesky(size, shape):
    assert type(size) == int # the size of the matrix
    assert type(shape) == float # shape parameter of distribution
    return 'LKJCorrCholesky({}, torch.tensor({}))'.format(size, shape)

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

def genprior(varname, prior_desc):
    assert type(varname) == str
    assert type(prior_desc) == list
    assert all(type(p)    == tuple and
               type(p[0]) == Prior and
               type(p[1]) == int
               for p in prior_desc)
    code = []

    # Sample each segment of a coefficient vector.
    for i, (prior, length) in enumerate(prior_desc):
        code.append(sample('{}_{}'.format(varname, i), gendist(prior.family, prior.arguments, [length], False)))

    # TODO: Optimisation -- avoid `torch.concat` when only sample is
    # drawn. (Binding the sampled value directly to `varname`.)

    if len(prior_desc) > 0:
        # Concatenate the segments to produce the final vector.
        code.append('{} = torch.cat([{}])'.format(varname, ', '.join('{}_{}'.format(varname, i) for i in range(len(prior_desc)))))
    else:
        code.append('{} = torch.tensor([])'.format(varname))

    return code


# Generates model code for a single group. More specifically, this
# generates code to sample group level priors and to accumulate the
# groups contribution to mu.
def gengroup(i, group):#, metadata, design_metadata, priors):
    assert type(i) == int # A unique int assigned to each group.
    assert type(group) == Group

    code = ['']
    code.append(comment('[{}] {}'.format(i, group.factor)))

    # The number of coefficients per level.
    M_i = len(group.coefs)

    # The number of levels.
    N_i = len(group.factor.levels)

    # This follows the names used in brms.
    code.append('M_{} = {} # Number of coeffs'.format(i, M_i))
    code.append('N_{} = {} # Number of levels'.format(i, N_i))

    code.append('assert type(Z_{}) == torch.Tensor'.format(i))
    code.append('assert Z_{}.shape == (N, M_{}) # N x {}'.format(i, i, M_i))
    code.append('assert type(J_{}) == torch.Tensor'.format(i))
    code.append('assert J_{}.shape == (N,)'.format(i))

    # Prior over coefficient scales.
    code.extend(genprior('sd_{}'.format(i), contig(group.sd_priors)))
    code.append('assert sd_{}.shape == (M_{},) # {}'.format(i, i, M_i))

    # Prior over a matrix of unscaled/uncorrelated coefficients. This
    # is similar to the brms generated Stan code. An alternative would
    # be to pass `torch.mm(torch.diag(sd_{}), L_{})` as the
    # `scale_tril` argument of a `MultivariateNormal`. Is there any
    # significant different between these two approaches?
    code.append(sample('z_{}'.format(i), gendist(getfamily('Normal'), [0., 1.], [M_i, N_i], batch=False)))
    code.append('assert z_{}.shape == (M_{}, N_{}) # {} x {}'.format(i, i, i, M_i, N_i))

    if group.corr_prior:
        # Model correlations between the coefficients.

        # This is guaranteed by the way the prior tree is built.
        assert M_i > 1

        # Prior over correlations.
        prior = group.corr_prior
        assert len(prior.arguments) == 1
        code.append(sample('L_{}'.format(i), lkj_corr_cholesky(M_i, shape=prior.arguments[0])))
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

def geninvlinkfn(linkfn, code):
    if linkfn == LinkFn.identity:
        return code
    elif linkfn == LinkFn.logit:
        return 'torch.sigmoid({})'.format(code)
    else:
        raise NotImplementedError('code generation for link function {} not implemented'.format(linkfn))


# TODO: I'm missing opportunities to vectorise here. Adjacent segments
# that share a family and differ only in parameters can be handled
# with a single `sample` statement with suitable parameters.

# e.g.
# contig(list('abb')) == [('a', 1), ('b', 2)]
def contig(xs):
    assert type(xs) == list or type(xs) == tuple # Though really more general than this.
    assert all(x is not None for x in xs) # Since None used as initial value of `cur`.
    cur = None
    segments = []
    for i, x in enumerate(xs):
        if x == cur:
            segments[-1][1].append(i) # Extend segment.
        else:
            cur = x
            segments.append((cur, [i])) # New segment.
    # Post-process.
    segments = [(x, len(ix)) for (x, ix) in segments]
    return segments


def genmodel(model):
    assert type(model) == Model
    num_groups = len(model.groups)

    body = []

    body.append('assert type(X) == torch.Tensor')
    body.append('N = X.shape[0]')

    # The number of columns in the design matrix.
    M = len(model.population.coefs)

    body.append('M = {}'.format(M))
    body.append('assert X.shape == (N, M)')

    # Population level
    # --------------------------------------------------

    # Prior over b. (The population level coefficients.)
    body.extend(genprior('b', contig(model.population.priors)))
    body.append('assert b.shape == (M,)')

    # Compute mu.
    body.append('mu = torch.mv(X, b)')

    # Group level
    # --------------------------------------------------
    for i, group in enumerate(model.groups):
        # Use 1 indexed groups to ease comparison of generate code
        # with code generated by brms.
        body.extend(gengroup(i+1, group))

    # Response
    # --------------------------------------------------

    # Sample from priors over the response distribution parameters
    # that aren't predicted from the data.
    for param, param_prior in zip(model.response.nonlocparams, model.response.priors):
        body.append(sample(param.name, gendist(param_prior.family, param_prior.arguments, [1], False)))


    # TODO: Optimisations (for numerical stability/perf.) are
    # available for some response family/link function pairs. (Though
    # maybe only for particular back-ends.) e.g. In Pyro `Bernoulli`
    # has a `logits` param, so it's possible to pass `mu` directly as
    # that.


    # TODO: This relies on the parameters defined in each Family
    # appearing in the same order as Pyro expects.
    def response_arg(param_name):
        if param_name == model.response.family.response.param:
            return geninvlinkfn(model.response.family.response.linkfn, 'mu')
        else:
            return '{}.expand(N)'.format(param_name)
    response_args = [response_arg(p.name) for p in model.response.family.params]

    body.append('with pyro.plate("obs", N):')
    body.append(indent(sample('y', gendist(model.response.family, response_args, shape=['N'], batch=True), 'y_obs')))

    # Values of interest that are not generated directly by sample
    # statements (such as the `b` vector) are returned from the model
    # so that they can be retrieved from the execution trace later.
    returned_params = (['b'] +
                       ['sd_{}'.format(i+1) for i in range(num_groups)] +
                       ['r_{}'.format(i+1) for i in range(num_groups)])
    retval =  '{{{}}}'.format(', '.join('\'{}\': {}'.format(p, p) for p in returned_params))
    body.append('return {}'.format(retval))

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
