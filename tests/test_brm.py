import numpy as np
from scipy.special import expit as sigmoid
import numpyro.handlers as numpyro
import pandas as pd
import pytest
import torch
from jax import random

import pyro.poutine as poutine
from brmp import define_model, brm, makedesc
from brmp.backend import data_from_numpy
from brmp.design import (Categorical, CategoricalCoding, Integral,
                         NumericCoding, RealValued, code_lengths, code_terms,
                         coef_names, dummy_df, make_column_lookup, makedata,
                         metadata_from_cols, metadata_from_df)
from brmp.family import (LKJ, Bernoulli, Binomial, HalfCauchy, HalfNormal,
                         Normal, StudentT, Poisson)
from brmp.fit import Samples
from brmp.formula import Formula, OrderedSet, Term, _1, allfactors, parse
from brmp.model import parameters, scalar_parameter_map, scalar_parameter_names
from brmp.model_pre import build_model_pre
from brmp.numpyro_backend import backend as numpyro_backend
from brmp.priors import Prior, build_prior_tree
from brmp.pyro_backend import backend as pyro_backend
from pyro.distributions import Independent


def assert_equal(a, b):
    assert type(a) == np.ndarray or type(a) == torch.Tensor
    assert type(a) == type(b)
    if type(a) == np.ndarray:
        assert (a == b).all()
    else:
        assert torch.equal(a, b)


default_params = dict(
    Normal=dict(loc=0., scale=1.),
    Cauchy=dict(loc=0., scale=1.),
    HalfCauchy=dict(scale=3.),
    HalfNormal=dict(scale=1.),
    LKJ=dict(eta=1.),
    Beta=dict(concentration1=1., concentration0=1.),
    StudentT=dict(df=3., loc=0., scale=1.),
)


# Makes list of columns metadata that includes an entry for every
# factor in `formula`. Any column not already in `cols` is assumed to
# be `RealValued`.
def expand_columns(formula, cols):
    lookup = make_column_lookup(cols)
    return [lookup.get(factor, RealValued(factor))
            for factor in allfactors(formula)]


codegen_cases = [
    # TODO: This (and similar examples below) can't be expressed with
    # the current parser. Is it useful to fix this (`y ~ -1`?), or can
    # these be dropped?
    # (Formula('y', [], []), [], [], ['sigma']),

    ('y ~ 1 + x', [], {}, Normal, [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    # Integer valued predictor.
    ('y ~ 1 + x', [Integral('x', min=0, max=10)], {}, Normal, [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    ('y ~ 1 + x1 + x2', [], {}, Normal, [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    ('y ~ x1:x2',
     [Categorical('x1', list('ab')), Categorical('x2', list('cd'))],
     {}, Normal, [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    # (Formula('y', [], [Group([], 'z', True)]), [Categorical('z', list('ab'))], [], ['sigma', 'z_1']),
    # Groups with fewer than two terms don't sample the (Cholesky
    # decomp. of the) correlation matrix.
    # (Formula('y', [], [Group([], 'z', True)]), [Categorical('z', list('ab'))], [], ['sigma', 'z_1']),
    ('y ~ 1 | z', [Categorical('z', list('ab'))], {}, Normal, [],
     [('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {})]),

    # Integers as categorical levels.
    ('y ~ 1 | z', [Categorical('z', [10, 20])], {}, Normal, [],
     [('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {})]),

    ('y ~ x | z', [Categorical('z', list('ab'))], {}, Normal, [],
     [('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {})]),

    ('y ~ x | z',
     [Categorical('x', list('ab')), Categorical('z', list('ab'))],
     {}, Normal, [],
     [('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {}),
      ('L_0', 'LKJ', {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 | z)', [Categorical('z', list('ab'))], {}, Normal, [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {}),
      ('L_0', 'LKJ', {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 || z)', [Categorical('z', list('ab'))], {}, Normal, [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 + x4 | z1) + (1 + x5 | z2)',
     [Categorical('z1', list('ab')), Categorical('z2', list('ab'))],
     {},
     Normal,
     [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {}),
      ('L_0', 'LKJ', {}),
      ('z_1', 'Normal', {}),
      ('sd_1_0', 'HalfCauchy', {}),
      ('L_1', 'LKJ', {})]),

    ('y ~ 1 | a:b',
     [Categorical('a', ['a1', 'a2']), Categorical('b', ['b1', 'b2'])],
     {},
     Normal,
     [],
     [('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {})]),

    # Custom priors.
    ('y ~ 1 + x1 + x2',
     [], {},
     Normal,
     [Prior(('b',), Normal(0., 100.))],
     [('b_0', 'Normal', {'loc': 0., 'scale': 100.}),
      ('sigma', 'HalfCauchy', {})]),

    ('y ~ 1 + x1 + x2',
     [], {},
     Normal,
     [Prior(('b', 'intercept'), Normal(0., 100.))],
     [('b_0', 'Normal', {'loc': 0., 'scale': 100.}),
      ('b_1', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    ('y ~ 1 + x1 + x2',
     [], {},
     Normal,
     [Prior(('b', 'x1'), Normal(0., 100.))],
     [('b_0', 'Cauchy', {}),
      ('b_1', 'Normal', {'loc': 0., 'scale': 100.}),
      ('b_2', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    ('y ~ 1',
     [], {},
     Normal,
     [Prior(('b',), StudentT(3., 0., 1.))],
     [('b_0', 'StudentT', {}),
      ('sigma', 'HalfCauchy', {})]),

    # Prior on coef of a factor.
    ('y ~ 1 + x',
     [Categorical('x', list('ab'))],
     {},
     Normal,
     [Prior(('b', 'x[b]'), Normal(0., 100.))],
     [('b_0', 'Cauchy', {}),
      ('b_1', 'Normal', {'loc': 0., 'scale': 100.}),
      ('sigma', 'HalfCauchy', {})]),

    # Prior on coef of an interaction.
    ('y ~ x1:x2',
     [Categorical('x1', list('ab')), Categorical('x2', list('cd'))],
     {},
     Normal,
     [Prior(('b', 'x1[b]:x2[c]'), Normal(0., 100.))],
     [('b_0', 'Cauchy', {}),
      ('b_1', 'Normal', {'loc': 0., 'scale': 100.}),
      ('b_2', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    # Prior on group level `sd` choice.
    ('y ~ 1 + x2 + x3 | x1',
     [Categorical('x1', list('ab'))],
     {},
     Normal,
     [Prior(('sd', 'x1', 'intercept'), HalfCauchy(4.))],
     [('sigma', 'HalfCauchy', {}),
      ('sd_0_0', 'HalfCauchy', {'scale': 4.}),
      ('sd_0_1', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('L_0', 'LKJ', {})]),

    ('y ~ 1 + x2 + x3 || x1',
     [Categorical('x1', list('ab'))],
     {},
     Normal,
     [Prior(('sd', 'x1', 'intercept'), HalfNormal(4.))],
     [('sigma', 'HalfCauchy', {}),
      ('sd_0_0', 'HalfNormal', {'scale': 4.}),
      ('sd_0_1', 'HalfCauchy', {}),
      ('z_0', 'Normal', {})]),

    ('y ~ 1 + x || a:b',
     [Categorical('a', ['a1', 'a2']), Categorical('b', ['b1', 'b2'])],
     {},
     Normal,
     [Prior(('sd', 'a:b', 'intercept'), HalfNormal(4.))],
     [('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfNormal', {'scale': 4.}),
      ('sd_0_1', 'HalfCauchy', {})]),

    # Prior on L.
    ('y ~ 1 + x2 | x1',
     [Categorical('x1', list('ab'))],
     {},
     Normal,
     [Prior(('cor',), LKJ(2.))],
     [('sigma', 'HalfCauchy', {}),
      ('sd_0_0', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('L_0', 'LKJ', {'eta': 2.})]),

    ('y ~ 1 + x | a:b',
     [Categorical('a', ['a1', 'a2']), Categorical('b', ['b1', 'b2'])],
     {},
     Normal,
     [Prior(('cor', 'a:b'), LKJ(2.))],
     [('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {}),
      ('L_0', 'LKJ', {'eta': 2.})]),

    # Prior on parameter of response distribution.
    ('y ~ x',
     [],
     {},
     Normal,
     [Prior(('resp', 'sigma'), HalfCauchy(4.))],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {'scale': 4.})]),

    # Custom response family.
    ('y ~ x',
     [],
     {},
     Normal(sigma=0.5),
     [],
     [('b_0', 'Cauchy', {})]),

    ('y ~ x',
     [Categorical('y', list('AB'))],
     {},
     Bernoulli,
     [],
     [('b_0', 'Cauchy', {})]),

    ('y ~ x',
     [Integral('y', min=0, max=1)],
     {},
     Bernoulli,
     [],
     [('b_0', 'Cauchy', {})]),

    ('y ~ x',
     [Integral('y', min=0, max=10)],
     {},
     Binomial(num_trials=10),
     [],
     [('b_0', 'Cauchy', {})]),

    ('y ~ 1 + x',
     [Integral('y', min=0, max=10), Integral('x', min=0, max=10)],
     {},
     Poisson,
     [],
     [('b_0', 'Cauchy', {})]),

    # Contrasts
    ('y ~ a',
     [Categorical('a', ['a1', 'a2'])],
     {'a': np.array([[-1, -1, -1], [1, 1, 1]])},
     Normal,
     [Prior(('b', 'a[custom.1]'), Normal(0., 1.))],
     [('b_0', 'Cauchy', {}),
      ('b_1', 'Normal', {}),
      ('b_2', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    ('y ~ a + (a | b)',
     [Categorical('a', ['a1', 'a2']), Categorical('b', ['b1', 'b2'])],
     {'a': np.array([[-1, -1, -1], [1, 1, 1]])},
     Normal, [
         Prior(('b', 'a[custom.1]'), Normal(0., 1.)),
         Prior(('sd', 'b', 'a[custom.0]'), HalfCauchy(4.))
     ],
     [('b_0', 'Cauchy', {}),
      ('b_1', 'Normal', {}),
      ('b_2', 'Cauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {'scale': 4.}),
      ('sd_0_1', 'HalfCauchy', {}),
      ('L_0', 'LKJ', {}),
      ('sigma', 'HalfCauchy', {})]),

]


# Map generic family names to backend specific names.
def pyro_family_name(name):
    return dict(LKJ='LKJCorrCholesky').get(name, name)


def numpyro_family_name(name):
    return dict(LKJ='LKJCholesky',
                Bernoulli='BernoulliProbs',
                Binomial='BinomialProbs').get(name, name)


@pytest.mark.parametrize('N', [1, 5])
@pytest.mark.parametrize('formula_str, non_real_cols, contrasts, family, priors, expected', codegen_cases)
def test_pyro_codegen(N, formula_str, non_real_cols, contrasts, family, priors, expected):
    # Make dummy data.
    formula = parse(formula_str)
    cols = expand_columns(formula, non_real_cols)
    # Generate the model from the column information rather than from
    # the metadata extracted from `df`. Since N is small, the metadata
    # extracted from `df` might loose information compared to the full
    # metadata derived from `cols` (e.g. levels of a categorical
    # column) leading to unexpected results. e.g. Missing levels might
    # cause correlations not to be modelled, even thought they ought
    # to be given the full metadata.
    metadata = metadata_from_cols(cols)
    desc = makedesc(formula, metadata, family, priors, code_lengths(contrasts))

    # Generate model function and data.
    modelfn = pyro_backend.gen(desc).fn

    df = dummy_df(cols, N, allow_non_exhaustive=True)
    data = data_from_numpy(pyro_backend, makedata(formula, df, metadata, contrasts))

    trace = poutine.trace(modelfn).get_trace(**data)

    # Check that y is correctly observed.
    y_node = trace.nodes['y']
    assert y_node['is_observed']
    assert type(y_node['fn']).__name__ == family.name
    assert_equal(y_node['value'], data['y_obs'])

    # Check sample sites.
    expected_sites = [site for (site, _, _) in expected]
    assert set(trace.stochastic_nodes) - {'obs'} == set(expected_sites)
    for (site, family_name, maybe_params) in expected:
        fn = unwrapfn(trace.nodes[site]['fn'])
        params = maybe_params or default_params[family_name]
        assert type(fn).__name__ == pyro_family_name(family_name)
        for (name, expected_val) in params.items():
            val = fn.__getattribute__(name)
            assert_equal(val, torch.tensor(expected_val).expand(val.shape))


def unwrapfn(fn):
    return unwrapfn(fn.base_dist) if type(fn) == Independent else fn


@pytest.mark.parametrize('N', [1, 5])
@pytest.mark.parametrize('formula_str, non_real_cols, contrasts, family, priors, expected', codegen_cases)
def test_numpyro_codegen(N, formula_str, non_real_cols, contrasts, family, priors, expected):
    # Make dummy data.
    formula = parse(formula_str)
    cols = expand_columns(formula, non_real_cols)
    metadata = metadata_from_cols(cols)
    desc = makedesc(formula, metadata, family, priors, code_lengths(contrasts))

    # Generate model function and data.
    modelfn = numpyro_backend.gen(desc).fn

    df = dummy_df(cols, N, allow_non_exhaustive=True)
    data = data_from_numpy(numpyro_backend, makedata(formula, df, metadata, contrasts))

    rng = random.PRNGKey(0)
    trace = numpyro.trace(numpyro.seed(modelfn, rng)).get_trace(**data)

    # Check that y is correctly observed.
    y_node = trace['y']
    assert y_node['is_observed']
    assert type(y_node['fn']).__name__ == numpyro_family_name(family.name)
    assert_equal(y_node['value'], data['y_obs'])

    # Check sample sites.
    expected_sites = [site for (site, _, _) in expected]
    sample_sites = [name for name, node in trace.items() if not node['is_observed']]
    assert set(sample_sites) == set(expected_sites)
    for (site, family_name, maybe_params) in expected:
        fn = trace[site]['fn']
        params = maybe_params or default_params[family_name]
        assert type(fn).__name__ == numpyro_family_name(family_name)
        for (name, expected_val) in params.items():
            if family_name == 'LKJ':
                assert name == 'eta'
                name = 'concentration'
            val = fn.__getattribute__(name)
            assert_equal(val._value, np.broadcast_to(expected_val, val.shape))


@pytest.mark.parametrize('formula_str, cols, expected', [
    ('y ~ 1 + x',
     [],
     lambda df, coef: coef('b_intercept') + df['x'] * coef('b_x')),
    ('y ~ a',
     [Categorical('a', ['a0', 'a1', 'a2'])],
     lambda df, coef: ((df['a'] == 'a0') * coef('b_a[a0]') +
                       (df['a'] == 'a1') * coef('b_a[a1]') +
                       (df['a'] == 'a2') * coef('b_a[a2]'))),
    ('y ~ 1 + a',
     [Categorical('a', ['a0', 'a1', 'a2'])],
     lambda df, coef: (coef('b_intercept') +
                       (df['a'] == 'a1') * coef('b_a[a1]') +
                       (df['a'] == 'a2') * coef('b_a[a2]'))),
    ('y ~ x1:x2',
     [],
     lambda df, coef: df['x1'] * df['x2'] * coef('b_x1:x2')),
    ('y ~ a:x',
     [Categorical('a', ['a0', 'a1'])],
     lambda df, coef: (((df['a'] == 'a0') * df['x'] * coef('b_a[a0]:x')) +
                       ((df['a'] == 'a1') * df['x'] * coef('b_a[a1]:x')))),
    ('y ~ 1 + x | a',
     [Categorical('a', ['a0', 'a1'])],
     lambda df, coef: ((df['a'] == 'a0') * (coef('r_a[a0,intercept]') + df['x'] * coef('r_a[a0,x]')) +
                       (df['a'] == 'a1') * (coef('r_a[a1,intercept]') + df['x'] * coef('r_a[a1,x]')))),
    ('y ~ 1 + x | a:b',
     [Categorical('a', ['a0', 'a1']), Categorical('b', ['b0', 'b1'])],
     lambda df, coef: (((df['a'] == 'a0') & (df['b'] == 'b0')) *
                       (coef('r_a:b[a0_b0,intercept]') + df['x'] * coef('r_a:b[a0_b0,x]')) +
                       ((df['a'] == 'a1') & (df['b'] == 'b0')) *
                       (coef('r_a:b[a1_b0,intercept]') + df['x'] * coef('r_a:b[a1_b0,x]')) +
                       ((df['a'] == 'a0') & (df['b'] == 'b1')) *
                       (coef('r_a:b[a0_b1,intercept]') + df['x'] * coef('r_a:b[a0_b1,x]')) +
                       ((df['a'] == 'a1') & (df['b'] == 'b1')) *
                       (coef('r_a:b[a1_b1,intercept]') + df['x'] * coef('r_a:b[a1_b1,x]')))),
    ('y ~ 1 + (x1 | a) + (x2 | b)',
     [Categorical('a', ['a0', 'a1']), Categorical('b', ['b0', 'b1'])],
     lambda df, coef: (coef('b_intercept') +
                       (df['a'] == 'a0') * df['x1'] * coef('r_a[a0,x1]') +
                       (df['a'] == 'a1') * df['x1'] * coef('r_a[a1,x1]') +
                       (df['b'] == 'b0') * df['x2'] * coef('r_b[b0,x2]') +
                       (df['b'] == 'b1') * df['x2'] * coef('r_b[b1,x2]'))),
])
@pytest.mark.parametrize('backend', [pyro_backend, numpyro_backend])
def test_mu_correctness(formula_str, cols, backend, expected):
    df = dummy_df(expand_columns(parse(formula_str), cols), 10)
    fit = brm(formula_str, df).prior(num_samples=1, backend=backend)
    # Pick out the one (and only) sample drawn.
    actual_mu = fit.fitted(what='linear')[0]
    # `expected` is assumed to return a data frame.
    expected_mu = expected(df, fit.get_scalar_param).to_numpy(np.float32)
    assert np.allclose(actual_mu, expected_mu)


@pytest.mark.parametrize('cols, family, expected', [
    ([],
     Normal,
     lambda mu: mu),
    ([Integral('y', min=0, max=1)],
     Bernoulli,
     lambda mu: sigmoid(mu)),
    ([Integral('y', min=0, max=5)],
     Binomial(num_trials=5),
     lambda mu: sigmoid(mu) * 5),
    ([Integral('y', min=0, max=5)],
     Poisson,
     lambda mu: np.exp(mu)),
])
@pytest.mark.parametrize('backend', [pyro_backend, numpyro_backend])
def test_expectation_correctness(cols, family, expected, backend):
    formula_str = 'y ~ 1 + x'
    df = dummy_df(expand_columns(parse(formula_str), cols), 10)
    fit = brm(formula_str, df, family=family).prior(num_samples=1, backend=backend)
    actual_expectation = fit.fitted(what='expectation')[0]
    # We assume (since it's tested elsewhere) that `mu` is computed
    # correctly by `fitted`. So given that, we check that `fitted`
    # computes the correct expectation.
    expected_expectation = expected(fit.fitted('linear')[0])
    assert np.allclose(actual_expectation, expected_expectation)


@pytest.mark.parametrize('N', [0, 5])
@pytest.mark.parametrize('backend', [pyro_backend, numpyro_backend])
@pytest.mark.parametrize('formula_str, non_real_cols, contrasts, family, priors, expected', codegen_cases)
def test_sampling_from_prior_smoke(N, backend, formula_str, non_real_cols, contrasts, family, priors, expected):
    formula = parse(formula_str)
    cols = expand_columns(formula, non_real_cols)
    metadata = metadata_from_cols(cols)  # Use full metadata for same reason given in comment in codegen test.
    desc = makedesc(formula, metadata, family, priors, code_lengths(contrasts))
    model = backend.gen(desc)
    df = dummy_df(cols, N, allow_non_exhaustive=True)
    data = data_from_numpy(backend, makedata(formula, df, metadata, contrasts))
    samples = backend.prior(data, model, num_samples=10, seed=None)
    assert type(samples) == Samples


@pytest.mark.parametrize('formula_str, non_real_cols, contrasts, family, priors, expected', codegen_cases)
@pytest.mark.parametrize('fitargs', [
    dict(backend=pyro_backend, num_samples=1, algo='prior'),
    dict(backend=numpyro_backend, num_samples=1, algo='prior'),
])
def test_parameter_shapes(formula_str, non_real_cols, contrasts, family, priors, expected, fitargs):
    # Make dummy data.
    N = 5
    formula = parse(formula_str)
    cols = expand_columns(formula, non_real_cols)
    df = dummy_df(cols, N, allow_non_exhaustive=True)

    # Define model, and generate a single posterior sample.
    metadata = metadata_from_cols(cols)
    model = define_model(formula_str, metadata, family, priors, contrasts).gen(fitargs['backend'])
    data = model.encode(df)
    fit = model.run_algo('prior', data, num_samples=1, seed=None)

    num_chains = fitargs.get('num_chains', 1)

    # Check parameter sizes.
    for parameter in parameters(fit.model_desc):
        expected_param_shape = parameter.shape
        samples = fit.get_param(parameter.name)
        # A single sample is collected by each chain for all cases.
        assert samples.shape == (num_chains,) + expected_param_shape
        samples_with_chain_dim = fit.get_param(parameter.name, True)
        assert samples_with_chain_dim.shape == (num_chains, 1) + expected_param_shape


def test_scalar_param_map_consistency():
    formula = parse('y ~ 1 + x1 + (1 + x2 + b | a) + (1 + x1 | a:b)')
    non_real_cols = [
        Categorical('a', ['a1', 'a2', 'a3']),
        Categorical('b', ['b1', 'b2', 'b3']),
    ]
    cols = expand_columns(formula, non_real_cols)
    desc = makedesc(formula, metadata_from_cols(cols), Normal, [], {})
    params = parameters(desc)
    spmap = scalar_parameter_map(desc)

    # Check that each entry in the map points to a unique parameter
    # position.
    param_and_indices_set = set(param_and_indices
                                for (_, param_and_indices) in spmap)
    assert len(param_and_indices_set) == len(spmap)

    # Ensure that we have enough entries in the map to cover all of
    # the scalar parameters. (The L_i parameters have a funny status.
    # We consider them to be parameters, but not scalar parameters.
    # This is not planned, rather things just evolved this way. It
    # does makes some sense though, since we usually look at R_i
    # instead.)
    num_scalar_params = sum(np.product(shape)
                            for name, shape in params
                            if not name.startswith('L_'))
    assert num_scalar_params == len(spmap)

    # Check that all indices are valid. (i.e. Within the shape of the
    # parameter.)
    for scalar_param_name, (param_name, indices) in spmap:
        ss = [shape for (name, shape) in params if name == param_name]
        assert len(ss) == 1
        param_shape = ss[0]
        assert len(indices) == len(param_shape)
        assert all(i < s for (i, s) in zip(indices, param_shape))


@pytest.mark.parametrize('formula_str, non_real_cols, contrasts, family, priors, expected', codegen_cases)
def test_scalar_parameter_names_smoke(formula_str, non_real_cols, contrasts, family, priors, expected):
    formula = parse(formula_str)
    cols = expand_columns(formula, non_real_cols)
    metadata = metadata_from_cols(cols)
    model = define_model(formula_str, metadata, family, priors, contrasts)
    names = scalar_parameter_names(model.desc)
    assert type(names) == list


@pytest.mark.parametrize('formula_str, non_real_cols, family, priors', [
    ('y ~ x', [], Bernoulli, []),
    ('y ~ x', [Integral('y', min=0, max=2)], Bernoulli, []),
    ('y ~ x', [Categorical('y', list('abc'))], Bernoulli, []),
    ('y ~ x', [Categorical('y', list('ab'))], Normal, []),
    ('y ~ x', [Integral('y', min=0, max=1)], Normal, []),
    ('y ~ x', [], Binomial(num_trials=1), []),
    ('y ~ x', [Integral('y', min=-1, max=1)], Binomial(num_trials=1), []),
    ('y ~ x',
     [Integral('y', min=0, max=3)],
     Binomial(num_trials=2),
     []),
    ('y ~ x', [Categorical('y', list('abc'))], Binomial(num_trials=1), []),
    ('y ~ x', [], Poisson, []),
])
def test_family_and_response_type_checks(formula_str, non_real_cols, family, priors):
    formula = parse(formula_str)
    cols = expand_columns(formula, non_real_cols)
    metadata = metadata_from_cols(cols)
    with pytest.raises(Exception, match='not compatible'):
        build_model_pre(formula, metadata, family, {})


@pytest.mark.parametrize('formula_str, non_real_cols, family, priors, expected_error', [
    ('y ~ x',
     [],
     Normal,
     [Prior(('resp', 'sigma'), Normal(0., 1.))],
     r'(?i)invalid prior'),
    ('y ~ x1 | x2',
     [Categorical('x2', list('ab'))],
     Normal,
     [Prior(('sd', 'x2'), Normal(0., 1.))],
     r'(?i)invalid prior'),
    ('y ~ 1 + x1 | x2',
     [Categorical('x2', list('ab'))],
     Normal,
     [Prior(('cor', 'x2'), Normal(0., 1.))],
     r'(?i)invalid prior'),
    ('y ~ x',
     [],
     Normal,
     [Prior(('b',), Bernoulli(.5))],
     r'(?i)invalid prior'),
    # This hasn't passed since I moved the family/response checks in
    # to the pre-model. The problem is that the support of the
    # Binomial response depends on its parameters which aren't fully
    # specified in this case, meaning that the family/reponse check
    # can't happen, and the prior test that ought to flag that a prior
    # is missing is never reached. It's not clear that a "prior
    # missing" error is the most helpful error to raise for this case,
    # and it's possible that having the family/response test suggest
    # that extra parameters ought to be specified is a better idea.
    # It's tricky to say though, since this case is a bit of a one
    # off, so figuring out a good general solution is tricky. Since
    # it's not clear how best to proceed, so I'll punt for now.
    pytest.param(
        'y ~ x',
        [Integral('y', 0, 1)],
        Binomial,
        [],
        r'(?i)prior missing', marks=pytest.mark.xfail),
])
def test_prior_checks(formula_str, non_real_cols, family, priors, expected_error):
    formula = parse(formula_str)
    cols = expand_columns(formula, non_real_cols)
    metadata = metadata_from_cols(cols)
    design_metadata = build_model_pre(formula, metadata, family, {})
    with pytest.raises(Exception, match=expected_error):
        build_prior_tree(design_metadata, priors)


@pytest.mark.parametrize('formula_str, df, metadata_cols, contrasts, expected', [
    # (Formula('y', [], []),
    #  pd.DataFrame(dict(y=[1, 2, 3])),
    #  dict(X=torch.tensor([[],
    #                       [],
    #                       []]),
    #       y_obs=torch.tensor([1., 2., 3.]))),
    ('y ~ 1',
     pd.DataFrame(dict(y=[1., 2., 3.])),
     None,
     {},
     dict(X=np.array([[1.],
                      [1.],
                      [1.]]),
          y_obs=np.array([1., 2., 3.]))),
    ('y ~ x',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=[4., 5., 6.])),
     None,
     {},
     dict(X=np.array([[4.],
                      [5.],
                      [6.]]),
          y_obs=np.array([1., 2., 3.]))),
    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=[4., 5., 6.])),
     None,
     {},
     dict(X=np.array([[1., 4.],
                      [1., 5.],
                      [1., 6.]]),
          y_obs=np.array([1., 2., 3.]))),
    ('y ~ x + 1',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=[4., 5., 6.])),
     None,
     {},
     dict(X=np.array([[1., 4.],
                      [1., 5.],
                      [1., 6.]]),
          y_obs=np.array([1., 2., 3.]))),

    ('y ~ x',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=pd.Categorical(list('AAB')))),
     None,
     {},
     dict(X=np.array([[1., 0.],
                      [1., 0.],
                      [0., 1.]]),
          y_obs=np.array([1., 2., 3.]))),
    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=pd.Categorical(list('AAB')))),
     None,
     {},
     dict(X=np.array([[1., 0.],
                      [1., 0.],
                      [1., 1.]]),
          y_obs=np.array([1., 2., 3.]))),
    ('y ~ x1 + x2',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x1=pd.Categorical(list('AAB')),
                       x2=pd.Categorical(list('ABC')))),
     None,
     {},
     dict(X=np.array([[1., 0., 0., 0.],
                      [1., 0., 1., 0.],
                      [0., 1., 0., 1.]]),
          y_obs=np.array([1., 2., 3.]))),

    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=pd.Categorical(list('ABC')))),
     None,
     {},
     dict(X=np.array([[1., 0., 0.],
                      [1., 1., 0.],
                      [1., 0., 1.]]),
          y_obs=np.array([1., 2., 3.]))),

    # (Formula('y', [], [Group([], 'x', True)]),
    #  pd.DataFrame(dict(y=[1, 2, 3],
    #                    x=pd.Categorical(list('ABC')))),
    #  dict(X=np.array([[],
    #                       [],
    #                       []]),
    #       y_obs=np.array([1., 2., 3.]),
    #       J_1=np.array([0, 1, 2]),
    #       Z_1=np.array([[],
    #                         [],
    #                         []]))),
    ('y ~ 1 + (1 + x1 | x2)',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x1=pd.Categorical(list('AAB')),
                       x2=pd.Categorical(list('ABC')))),
     None,
     {},
     dict(X=np.array([[1.],
                      [1.],
                      [1.]]),
          y_obs=np.array([1., 2., 3.]),
          J_0=np.array([0, 1, 2]),
          Z_0=np.array([[1., 0.],
                        [1., 0.],
                        [1., 1.]]))),

    # The matches brms modulo 0 vs. 1 based indexing.
    ('y ~ 1 | a:b:c',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       a=pd.Categorical([0, 0, 1]),
                       b=pd.Categorical([2, 1, 0]),
                       c=pd.Categorical([0, 1, 2]))),
     None,
     {},
     dict(X=np.array([[], [], []]),
          y_obs=np.array([1., 2., 3.]),
          J_0=np.array([1, 0, 2]),
          Z_0=np.array([[1.], [1.], [1.]]))),

    # Interactions
    # --------------------------------------------------
    ('y ~ x1:x2',
     pd.DataFrame(dict(y=[1., 2., 3., 4.],
                       x1=pd.Categorical(list('ABAB')),
                       x2=pd.Categorical(list('CCDD')))),
     None,
     {},
     #                     AC  BC  AD  BD
     dict(X=np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]]),
          y_obs=np.array([1., 2., 3., 4.]))),

    ('y ~ 1 + x1:x2',
     pd.DataFrame(dict(y=[1., 2., 3., 4.],
                       x1=pd.Categorical(list('ABAB')),
                       x2=pd.Categorical(list('CCDD')))),
     None,
     {},
     #                     1   D   BC  BD
     dict(X=np.array([[1., 0., 0., 0.],
                      [1., 0., 1., 0.],
                      [1., 1., 0., 0.],
                      [1., 1., 0., 1.]]),
          y_obs=np.array([1., 2., 3., 4.]))),

    ('y ~ 1 + x1 + x2 + x1:x2',
     pd.DataFrame(dict(y=[1., 2., 3., 4.],
                       x1=pd.Categorical(list('ABAB')),
                       x2=pd.Categorical(list('CCDD')))),
     None,
     {},
     #                     1   B   D   BD
     dict(X=np.array([[1., 0., 0., 0.],
                      [1., 1., 0., 0.],
                      [1., 0., 1., 0.],
                      [1., 1., 1., 1.]]),
          y_obs=np.array([1., 2., 3., 4.]))),

    # real-real
    ('y ~ x1:x2',
     pd.DataFrame(dict(y=[1., 2., 3., 4.],
                       x1=np.array([1., 2., 1., 2.]),
                       x2=np.array([-10., 0., 10., 20.]))),
     None,
     {},
     dict(X=np.array([[-10.],
                      [0.],
                      [10.],
                      [40.]]),
          y_obs=np.array([1., 2., 3., 4.]))),

    # real-int
    ('y ~ x1:x2',
     pd.DataFrame(dict(y=[1., 2., 3., 4.],
                       x1=np.array([1., 2., 1., 2.]),
                       x2=np.array([-10, 0, 10, 20]))),
     None,
     {},
     dict(X=np.array([[-10.],
                      [0.],
                      [10.],
                      [40.]]),
          y_obs=np.array([1., 2., 3., 4.]))),

    # real-categorical
    ('y ~ x1:x2',
     pd.DataFrame(dict(y=[1., 2., 3., 4.],
                       x1=np.array([1., 2., 3., 4.]),
                       x2=pd.Categorical(list('ABAB')))),
     None,
     {},
     dict(X=np.array([[1., 0.],
                      [0., 2.],
                      [3., 0.],
                      [0., 4.]]),
          y_obs=np.array([1., 2., 3., 4.]))),

    # This example is taken from here:
    # https://patsy.readthedocs.io/en/latest/R-comparison.html
    ('y ~ a:x + a:b',
     pd.DataFrame(dict(y=[1., 2., 3., 4.],
                       a=pd.Categorical(list('ABAB')),
                       b=pd.Categorical(list('CCDD')),
                       x=np.array([1., 2., 3., 4.]))),
     None,
     {},
     dict(X=np.array([[1., 0., 0., 0., 1., 0.],
                      [0., 1., 0., 0., 0., 2.],
                      [0., 0., 1., 0., 3., 0.],
                      [0., 0., 0., 1., 0., 4.]]),
          y_obs=np.array([1., 2., 3., 4.]))),

    # Integer-valued Factors
    # --------------------------------------------------
    ('y ~ x1 + x2',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x1=[4, 5, 6],
                       x2=[7., 8., 9.])),
     None,
     {},
     dict(X=np.array([[4., 7.],
                      [5., 8.],
                      [6., 9.]]),
          y_obs=np.array([1., 2., 3.]))),

    # Categorical Response
    # --------------------------------------------------
    ('y ~ x',
     pd.DataFrame(dict(y=pd.Categorical(list('AAB')),
                       x=[1., 2., 3.])),
     None,
     {},
     dict(X=np.array([[1.],
                      [2.],
                      [3.]]),
          y_obs=np.array([0., 0., 1.]))),

    # Contrasts
    # --------------------------------------------------
    ('y ~ a',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       a=pd.Categorical(['a1', 'a1', 'a2']))),
     None,
     {'a': np.array([[-1], [1]])},
     dict(X=np.array([[-1.],
                      [-1.],
                      [1.]]),
          y_obs=np.array([1., 2., 3.]))),

    ('y ~ a',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       a=pd.Categorical(['a1', 'a1', 'a2']))),
     [RealValued('y'), Categorical('a', levels=['a0', 'a1', 'a2'])],
     {'a': np.array([[0], [-1], [1]])},
     dict(X=np.array([[-1.],
                      [-1.],
                      [1.]]),
          y_obs=np.array([1., 2., 3.]))),

    ('y ~ a',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       a=pd.Categorical(['a1', 'a1', 'a2']))),
     None,
     {'a': np.array([[-1, -2], [0, 1]])},
     dict(X=np.array([[-1., -2.],
                      [-1., -2.],
                      [0., 1.]]),
          y_obs=np.array([1., 2., 3.]))),

    ('y ~ 1 + a + b + a:b',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       a=pd.Categorical(['a1', 'a1', 'a2']),
                       b=pd.Categorical(['b1', 'b2', 'b2']))),
     None,
     {'a': np.array([[-1], [1]]), 'b': np.array([[2], [3]])},
     dict(X=np.array([[1., -1., 2., -2.],
                      [1., -1., 3., -3.],
                      [1., 1., 3., 3.]]),
          y_obs=np.array([1., 2., 3.]))),

    ('y ~ 1 + (a | b)',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       a=pd.Categorical(['a1', 'a1', 'a2']),
                       b=pd.Categorical(['b1', 'b2', 'b2']))),
     None,
     {'a': np.array([[-1], [1]])},
     dict(X=np.array([[1.],
                      [1.],
                      [1.]]),
          Z_0=np.array([[-1.],
                        [-1.],
                        [1.]]),
          J_0=np.array([0, 1, 1]),
          y_obs=np.array([1., 2., 3.]))),

])
def test_designmatrix(formula_str, df, metadata_cols, contrasts, expected):
    metadata = metadata_from_cols(metadata_cols) if metadata_cols is not None else metadata_from_df(df)
    data = makedata(parse(formula_str), df, metadata, contrasts)
    assert set(data.keys()) == set(expected.keys())
    for k in expected.keys():
        assert data[k].dtype == expected[k].dtype
        assert_equal(data[k], expected[k])


@pytest.mark.parametrize('formula_str, expected_formula', [
    ('y ~ 1', Formula('y', OrderedSet(_1), [])),
    ('y ~ 1 + x', Formula('y', OrderedSet(_1, Term(OrderedSet('x'))), [])),
    ('y ~ x + x', Formula('y', OrderedSet(Term(OrderedSet('x'))), [])),
    ('y ~ x1 : x2', Formula('y', OrderedSet(Term(OrderedSet('x1', 'x2'))), [])),
    ('y ~ (x1 + x2) : x3',
     Formula('y',
             OrderedSet(Term(OrderedSet('x1', 'x3')),
                        Term(OrderedSet('x2', 'x3'))),
             [])),
])
def test_parser(formula_str, expected_formula):
    formula = parse(formula_str)
    assert formula == expected_formula


def mkcat(factor, num_levels):
    return Categorical(factor, levels=['{}{}'.format(factor, i + 1) for i in range(num_levels)])


@pytest.mark.parametrize('formula_str, non_real_cols, expected_coding', [
    ('y ~ 1', [],
     [
         []  # intercept
     ]),
    ('y ~ x',
     [mkcat('x', 2)],
     [
         [CategoricalCoding('x', False)]
     ]),
    ('y ~ 1 + x',
     [mkcat('x', 2)],
     [
         [],
         [CategoricalCoding('x', True)]
     ]),
    ('y ~ a:b',
     [mkcat('a', 2), mkcat('b', 2)],
     [
         [CategoricalCoding('a', False), CategoricalCoding('b', False)]  # a:b
     ]),
    ('y ~ 1 + a:b',
     [mkcat('a', 2), mkcat('b', 2)],
     [
         [],
         [CategoricalCoding('b', True)],  # b-
         [CategoricalCoding('a', True), CategoricalCoding('b', False)]  # a-:b
     ]),
    ('y ~ 1 + a + a:b',
     [mkcat('a', 2), mkcat('b', 2)],
     [
         [],  # Intercept
         [CategoricalCoding('a', True)],  # a-
         [CategoricalCoding('a', False), CategoricalCoding('b', True)]  # a:b-
     ]),
    ('y ~ 1 + b + a:b',
     [mkcat('a', 2), mkcat('b', 2)],
     [
         [],  # Intercept
         [CategoricalCoding('b', True)],  # b-
         [CategoricalCoding('a', True), CategoricalCoding('b', False)]  # a-:b
     ]),
    ('y ~ 1 + a + b + a:b',
     [mkcat('a', 2), mkcat('b', 2)],
     [
         [],  # Intercept
         [CategoricalCoding('a', True)],  # a-
         [CategoricalCoding('b', True)],  # b-
         [CategoricalCoding('a', True), CategoricalCoding('b', True)]  # a-:b-
     ]),
    ('y ~ a:b + a:b:c',
     [mkcat('a', 2), mkcat('b', 2), mkcat('c', 2)],
     [
         [CategoricalCoding('a', False), CategoricalCoding('b', False)],  # a:b
         [CategoricalCoding('a', False), CategoricalCoding('b', False), CategoricalCoding('c', True)],  # a:b:c-
     ]),
    # This is based on an example in the Patsy docs:
    # https://patsy.readthedocs.io/en/latest/formulas.html#from-terms-to-matrices
    ('y ~ 1 + x1:x2 + a:b + b + x1:a:b + a + x2:a:x1',
     [mkcat('a', 2), mkcat('b', 2)],
     [
         [],
         [CategoricalCoding('b', True)],
         [CategoricalCoding('a', True)],
         [CategoricalCoding('a', True), CategoricalCoding('b', True)],
         [NumericCoding('x1'), NumericCoding('x2')],
         [NumericCoding('x2'), CategoricalCoding('a', True), NumericCoding('x1')],
         [NumericCoding('x1'), CategoricalCoding('a', False), CategoricalCoding('b', False)],
     ])
])
def test_coding(formula_str, non_real_cols, expected_coding):
    formula = parse(formula_str)
    cols = expand_columns(formula, non_real_cols)
    metadata = metadata_from_cols(cols)
    assert code_terms(formula.terms, metadata) == expected_coding


@pytest.mark.parametrize('formula_str, non_real_cols, expected_names', [
    # This is based on an example in the Patsy docs:
    # https://patsy.readthedocs.io/en/latest/formulas.html#from-terms-to-matrices
    ('y ~ 1 + x1:x2 + a:b + b + x1:a:b + a + x2:a:x1',
     [mkcat('a', 2), mkcat('b', 2)],
     ['intercept', 'b[b2]', 'a[a2]', 'a[a2]:b[b2]', 'x1:x2', 'x2:a[a2]:x1',
      'x1:a[a1]:b[b1]', 'x1:a[a2]:b[b1]', 'x1:a[a1]:b[b2]', 'x1:a[a2]:b[b2]'])
])
def test_coef_names(formula_str, non_real_cols, expected_names):
    formula = parse(formula_str)
    cols = expand_columns(formula, non_real_cols)
    metadata = metadata_from_cols(cols)
    assert coef_names(formula.terms, metadata, {}) == expected_names


@pytest.mark.parametrize('fitargs', [
    lambda S: dict(backend=pyro_backend, algo='prior', num_samples=S),
    lambda S: dict(backend=pyro_backend, iter=S, warmup=0),
    lambda S: dict(backend=pyro_backend, iter=S // 2, num_chains=2, warmup=0),
    lambda S: dict(backend=pyro_backend, algo='svi', iter=1, num_samples=S),
    lambda S: dict(backend=pyro_backend, algo='svi', iter=1, num_samples=S, subsample_size=1),
    lambda S: dict(backend=numpyro_backend, algo='prior', num_samples=S),
    lambda S: dict(backend=numpyro_backend, iter=S, warmup=0),
    lambda S: dict(backend=numpyro_backend, iter=S // 2, num_chains=2, warmup=0),
])
@pytest.mark.parametrize('formula_str, non_real_cols, family, contrasts', [
    ('y ~ 1 + a + x + (1 | b)',
     [Categorical('a', ['a0', 'a1']), Categorical('b', ['b0', 'b1'])],
     Normal,
     {}),
])
def test_marginals_fitted_smoke(fitargs, formula_str, non_real_cols, family, contrasts):
    N = 10
    S = 4
    formula = parse(formula_str)
    cols = expand_columns(formula, non_real_cols)
    df = dummy_df(cols, N)
    model = brm(formula_str, df, family, [], contrasts)
    fit = model.fit(**fitargs(S))
    # Sanity check output for `marginals`.
    arr = fit.marginals().array
    num_coefs = len(scalar_parameter_names(fit.model_desc))
    assert arr.shape == (num_coefs, 9)  # num coefs x num stats
    # Don't check finiteness of n_eff and r_hat, which are frequently
    # nan with few samples
    assert np.all(np.isfinite(arr[:, :-2]))

    # Sanity check output of `fitted`.
    def chk(arr, expected_shape):
        assert np.all(np.isfinite(arr))
        assert arr.shape == expected_shape
    chk(fit.fitted(), (S, N))
    chk(fit.fitted('linear'), (S, N))
    chk(fit.fitted('response'), (S, N))
    chk(fit.fitted('sample'), (S, N))
    chk(fit.fitted(data=dummy_df(cols, N)), (S, N))


# Testing with N2=1 ensures that for a model with categorical
# factors/columns, `new_data` can only include a proper subset of the
# available levels. Such data must be coded using the original
# metadata in order ensure to be compatible with the model, and this
# test exercises that.
@pytest.mark.parametrize('N2', [1, 8])
def test_fitted_on_new_data(N2):
    S = 4
    N = 10
    formula_str = 'y ~ 1 + a'
    # Using this contrast means `a` is coded as two columns rather
    # than (the default) one. Because of this, it's crucial that
    # `fitted` uses the contrast when coding *new data*. This test
    # would fail if that didn't happen.
    contrasts = {'a': np.array([[-1, -1], [1, 1]])}
    cols = expand_columns(parse(formula_str), [Categorical('a', ['a0', 'a1'])])
    df = dummy_df(cols, N)
    fit = brm(formula_str, df, Normal, contrasts=contrasts).fit(iter=S, backend=pyro_backend)
    new_data = dummy_df(cols, N2, allow_non_exhaustive=True)
    arr = fit.fitted(data=new_data)
    assert np.all(np.isfinite(arr))
    assert arr.shape == (S, N2)


@pytest.mark.parametrize('fitargs', [
    dict(backend=pyro_backend, algo='prior', num_samples=10),
    dict(backend=pyro_backend, algo='nuts', iter=10, warmup=0),
    dict(backend=pyro_backend, algo='svi', iter=10, num_samples=10),
    dict(backend=numpyro_backend, algo='prior', num_samples=10),
    dict(backend=numpyro_backend, algo='nuts', iter=10, warmup=0),
])
def test_rng_seed(fitargs):
    df = pd.DataFrame({'y': [0., 0.1, 0.2]})
    model = brm('y ~ 1', df)
    fit0 = model.fit(seed=0, **fitargs)
    fit1 = model.fit(seed=0, **fitargs)
    fit2 = model.fit(seed=1, **fitargs)
    assert (fit0.fitted() == fit1.fitted()).all()
    assert not (fit1.fitted() == fit2.fitted()).all()
    fitted0 = fit0.fitted(what='sample', seed=0)
    fitted1 = fit0.fitted(what='sample', seed=0)
    fitted2 = fit0.fitted(what='sample', seed=1)
    assert (fitted0 == fitted1).all()
    assert not (fitted1 == fitted2).all()
