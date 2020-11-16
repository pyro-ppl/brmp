import re

from brmp.backend import Assets
from brmp.family import Family, LinkFn, Normal, args, free_param_names
from brmp.model import Group, ModelDesc
from brmp.utils import traceback_generated


def gen_expanded_scalar(val, shape):
    assert type(val) in [float, int]
    return 'torch.tensor({}).expand({})'.format(val, ', '.join(str(dim) for dim in shape))


def gendist(family, args, shape, batch):
    assert type(family) == Family
    assert type(args) == list
    assert len(args) == len(family.params)
    # Floats and ints are expanded to `shape`, string are assumed to
    # be literal code.
    assert all(type(arg) in [float, int, str] for arg in args)
    assert type(shape) == list
    # Dimensions are ints (when statically known) or strings (when
    # know at run-time only).
    assert all(type(dim) in [int, str] for dim in shape)
    assert type(batch) == bool

    args_code = [arg if type(arg) == str else gen_expanded_scalar(arg, shape) for arg in args]
    out = '{}({})'.format(family.name, ', '.join(args_code))

    # It is sufficient for present purposes that this function only
    # handles the cases where either all dims are *event* dims (e.g.
    # when sampling a vector of coefs) or where all dims are *batch*
    # dims (e.g. sampling a vector/matrix of response variables, which
    # can only be scalar at present).
    if not batch:
        out = out + '.to_event({})'.format(len(shape))
    return out


def gen_response_dist(model, vectorize=False):
    shape = ['S', 'N'] if vectorize else ['N']

    # TODO: Optimisations (for numerical stability/perf.) are
    # available for some response family/link function pairs. (Though
    # maybe only for particular back-ends.) e.g. In Pyro `Bernoulli`
    # has a `logits` param, so it's possible to pass `mu` directly as
    # that.

    # TODO: This relies on the parameters defined in each Family
    # appearing in the same order as Pyro expects.
    def response_arg(param):
        if param.name == model.response.family.link.param:
            return geninvlinkbody(model.response.family.link.fn, 'mu')
        elif param.value is not None:
            return param.value  # Will be made into an expanded tensor by `gendist`.
        else:
            return '{}.expand({})'.format(param.name, ', '.join(shape))

    response_args = [response_arg(p) for p in model.response.family.params]
    return gendist(model.response.family, response_args, shape=shape, batch=True)


def lkj_corr_cholesky(size, shape):
    assert type(size) == int  # the size of the matrix
    assert type(shape) == float  # shape parameter of distribution
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
    assert all(type(p) == tuple and
               type(p[0]) == Family and
               type(p[1]) == int
               for p in prior_desc)
    code = []

    # Sample each segment of a coefficient vector.
    for i, (prior, length) in enumerate(prior_desc):
        code.append(sample('{}_{}'.format(varname, i), gendist(prior, args(prior), [length], False)))

    if len(prior_desc) == 0:
        code.append('{} = torch.tensor([])'.format(varname))
    elif len(prior_desc) == 1:
        code.append(f'{varname} = {varname}_0')
    else:
        # Concatenate the segments to produce the final vector.
        varname_coefs = ", ".join(f'{varname}_{i}' for i in range(len(prior_desc)))
        code.append(f'{varname} = torch.cat([{varname_coefs}])')

    return code


# Generates model code for a single group. More specifically, this
# generates code to sample group level priors and to accumulate the
# groups contribution to mu.
def gengroup(i, group):
    assert type(i) == int  # A unique int assigned to each group.
    assert type(group) == Group

    cmt = comment('Group {}: factor={}'.format(i, ':'.join(group.columns)))
    code = ['', cmt]
    mu_code = [cmt]

    # The number of coefficients per level.
    M_i = len(group.coefs)

    # The number of levels.
    N_i = len(group.levels)

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
    code.append(sample('z_{}'.format(i), gendist(Normal, [0., 1.], [M_i, N_i], batch=False)))
    code.append('assert z_{}.shape == (M_{}, N_{}) # {} x {}'.format(i, i, i, M_i, N_i))

    if group.corr_prior:
        # Model correlations between the coefficients.

        # This is guaranteed by the way the prior tree is built.
        assert M_i > 1

        # Prior over correlations.
        prior = group.corr_prior
        assert len(args(prior)) == 1
        code.append(sample('L_{}'.format(i), lkj_corr_cholesky(M_i, shape=args(prior)[0])))
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

    # XXX: This allocates a large intermediate tensor `r_1[J_1]`.
    # An alternative might be to iterate over N_i levels and use
    # scatter_add to add that level's contribution to the rows
    # in mu that belong to that level.

    mu_code.append(f'mu = mu + torch.sum(Z_{i} * r_{i}[J_{i}], 1)')

    return code, mu_code


def geninvlinkbody(linkfn, code):
    if linkfn == LinkFn.identity:
        return code
    elif linkfn == LinkFn.logit:
        return 'torch.sigmoid({})'.format(code)
    elif linkfn == LinkFn.log:
        return 'torch.exp({})'.format(code)
    else:
        raise NotImplementedError('code generation for link function {} not implemented'.format(linkfn))


# TODO: Re-evaluate whether it really makes sense to have these
# implemented by each back end. An alternative is to implement link
# functions / response expectations once, as functions which operate
# on parameters represented as numpy arrays. (Since I imagining that
# each back end will come with the ability to map its parameter values
# to numpy arrays.) This would avoid having to so something like this
# for every backend, and would have the advantage of not coming from
# generated code. On the downside, it might mean duplicating some
# logic, e.g. for parameter shapes.
def geninvlinkfn(model):
    body = geninvlinkbody(model.response.family.link.fn, 'x')
    return '\n'.join(method('invlink', ['x'], ['return {}'.format(body)]))


def gen_response_fn(model, mode):
    assert mode in ['expectation', 'sample']
    distcode = 'dist.{}'.format(gen_response_dist(model, vectorize=True))
    args = free_param_names(model.response.family)
    retvalfmt = 'pyro.sample("y", {})' if mode == 'sample' else '{}.mean'
    retval = retvalfmt.format(distcode)
    body = ["S, N = mu.shape",
            "return {}".format(retval)]
    return '\n'.join(method('expected_response', args, body))


# TODO: I'm missing opportunities to vectorise here. Adjacent segments
# that share a family and differ only in parameters can be handled
# with a single `sample` statement with suitable parameters.

# e.g.
# contig(list('abb')) == [('a', 1), ('b', 2)]
def contig(xs):
    assert type(xs) == list or type(xs) == tuple  # Though really more general than this.
    assert all(x is not None for x in xs)  # Since None used as initial value of `cur`.
    cur = None
    segments = []
    for i, x in enumerate(xs):
        if x == cur:
            segments[-1][1].append(i)  # Extend segment.
        else:
            cur = x
            segments.append((cur, [i]))  # New segment.
    # Post-process.
    segments = [(x, len(ix)) for (x, ix) in segments]
    return segments


def genmodel(model):
    assert type(model) == ModelDesc
    num_groups = len(model.groups)

    body = []

    body.append('assert mode == "full" or mode == "prior_and_mu" or mode == "prior_only"')
    body.append('assert (subsample is None) == (dfN is None)')  # Expect both or neither.

    body.append('assert type(X) == torch.Tensor')
    body.append('N = X.shape[0]')

    body.append('if dfN is None:')
    body.append(indent('dfN = N'))
    body.append('else:')
    body.append(indent('assert len(subsample) == N'))

    # The number of columns in the design matrix.
    M = len(model.population.coefs)

    body.append('M = {}'.format(M))
    body.append('assert X.shape == (N, M)')
    body.append('')

    # Population level
    # --------------------------------------------------

    # Prior over b. (The population level coefficients.)
    body.extend(genprior('b', contig(model.population.priors)))
    body.append('assert b.shape == (M,)')

    # Group level
    # --------------------------------------------------
    mu_code = []
    for i, group in enumerate(model.groups):
        grp_code, grp_mu_code = gengroup(i, group)
        body.extend(grp_code)
        mu_code.extend(grp_mu_code)

    # Compute mu.
    body.append('')
    body.append('if mode == "prior_only":')
    body.append(indent('mu = None'))
    body.append('else:')
    body.append(indent('mu = torch.mv(X, b)'))
    body.extend(indent(line) for line in mu_code)
    body.append('')

    # Response
    # --------------------------------------------------

    # Sample from priors over the response distribution parameters
    # that aren't predicted from the data.
    for param, param_prior in zip(model.response.nonlocparams, model.response.priors):
        body.append(sample(param.name, gendist(param_prior, args(param_prior), [1], False)))

    body.append('if mode == "full":')
    body.append(indent('with pyro.plate("obs", dfN, subsample=subsample):'))
    body.append(indent(indent(sample('y', gen_response_dist(model), 'y_obs'))))

    # Values of interest that are not generated directly by sample
    # statements (such as the `b` vector) are returned from the model
    # so that they can be retrieved from the execution trace later.
    returned_params = (['mu', 'b'] +
                       ['sd_{}'.format(i) for i in range(num_groups)] +
                       ['r_{}'.format(i) for i in range(num_groups)])
    retval = '{{{}}}'.format(', '.join('\'{}\': {}'.format(p, p) for p in returned_params))
    body.append('')
    body.append('return {}'.format(retval))

    params = (['X'] +
              ['Z_{}'.format(i) for i in range(num_groups)] +
              ['J_{}'.format(i) for i in range(num_groups)] +
              ['y_obs=None', 'dfN=None', 'subsample=None', 'mode="full"'])
    return '\n'.join(method('model', params, body))


def eval_method(code):
    match = re.search(r'^def +(\w+)\(', code)
    assert match is not None
    method_name = match.group(1)
    import torch  # noqa: F401
    import pyro  # noqa: F401
    import pyro.distributions as dist  # noqa: F401
    g = locals()
    exec(code, g)
    return traceback_generated(fn=g[method_name], code=code)


def gen(model_desc):
    assert type(model_desc) == ModelDesc
    code = genmodel(model_desc)
    fn = eval_method(code)
    inv_link_code = geninvlinkfn(model_desc)
    inv_link_fn = eval_method(inv_link_code)
    expected_response_code = gen_response_fn(model_desc, mode='expectation')
    expected_response_fn = eval_method(expected_response_code)
    sample_response_code = gen_response_fn(model_desc, mode='sample')
    sample_response_fn = eval_method(sample_response_code)
    return Assets(fn, code,
                  inv_link_fn, inv_link_code,
                  expected_response_fn, expected_response_code,
                  sample_response_fn, sample_response_code)
