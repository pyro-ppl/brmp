import os
import pytest

import numpy as np
import torch
import pandas as pd

import pyro.poutine as poutine
from pyro.distributions import Independent

from jax import random
import numpyro.handlers as numpyro

from pyro.contrib.brm import brm, defm
from pyro.contrib.brm.formula import parse, Formula, _1, Term, OrderedSet, allfactors
from pyro.contrib.brm.design import dummy_design, Categorical, RealValued, Integral, makedata, make_metadata_lookup, designmatrices_metadata, CodedFactor, categorical_coding, dummy_df
from pyro.contrib.brm.priors import prior, PriorEdit, get_response_prior, build_prior_tree
from pyro.contrib.brm.family import Family, getfamily, FAMILIES, Type, apply
from pyro.contrib.brm.model import build_model, parameters
from pyro.contrib.brm.fit import marginals, fitted
from pyro.contrib.brm.pyro_backend import backend as pyro_backend
from pyro.contrib.brm.numpyro_backend import backend as numpyro_backend

from tests.common import assert_equal

default_params = dict(
    Normal     = dict(loc=0., scale=1.),
    Cauchy     = dict(loc=0., scale=1.),
    HalfCauchy = dict(scale=3.),
    HalfNormal = dict(scale=1.),
    LKJ        = dict(eta=1.),
)

def build_metadata(formula, metadata):
    # Helper that assumes that any factors not mentioned in the
    # metadata given with the test definition are real-valued.
    default_metadata = make_metadata_lookup([RealValued(factor) for factor in allfactors(formula)])
    return dict(default_metadata, **make_metadata_lookup(metadata))

codegen_cases = [
    # TODO: This (and similar examples below) can't be expressed with
    # the current parser. Is it useful to fix this (`y ~ -1`?), or can
    # these be dropped?
    #(Formula('y', [], []), [], [], ['sigma']),

    ('y ~ 1 + x', [], getfamily('Normal'), [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    # Integer valued predictor.
    ('y ~ 1 + x', [Integral('x', min=0, max=10)], getfamily('Normal'), [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    ('y ~ 1 + x1 + x2', [], getfamily('Normal'), [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    ('y ~ x1:x2',
     [Categorical('x1', list('ab')), Categorical('x2', list('cd'))],
     getfamily('Normal'), [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    #(Formula('y', [], [Group([], 'z', True)]), [Categorical('z', list('ab'))], [], ['sigma', 'z_1']),
    # Groups with fewer than two terms don't sample the (Cholesky
    # decomp. of the) correlation matrix.
    #(Formula('y', [], [Group([], 'z', True)]), [Categorical('z', list('ab'))], [], ['sigma', 'z_1']),
    ('y ~ 1 | z', [Categorical('z', list('ab'))], getfamily('Normal'), [],
     [('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {})]),

    ('y ~ x | z', [Categorical('z', list('ab'))], getfamily('Normal'), [],
     [('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 | z)', [Categorical('z', list('ab'))], getfamily('Normal'), [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {}),
      ('L_0', 'LKJ', {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 || z)', [Categorical('z', list('ab'))], getfamily('Normal'), [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 + x4 | z1) + (1 + x5 | z2)',
     [Categorical('z1', list('ab')), Categorical('z2', list('ab'))],
     getfamily('Normal'),
     [],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('sd_0_0', 'HalfCauchy', {}),
      ('L_0', 'LKJ', {}),
      ('z_1', 'Normal', {}),
      ('sd_1_0', 'HalfCauchy', {}),
      ('L_1', 'LKJ', {})]),

    # Custom priors.
    ('y ~ 1 + x1 + x2',
     [],
     getfamily('Normal'),
     [PriorEdit(('b',), prior('Normal', [0., 100.]))],
     [('b_0', 'Normal', {'loc': 0., 'scale': 100.}),
      ('sigma', 'HalfCauchy', {})]),

    ('y ~ 1 + x1 + x2',
     [],
     getfamily('Normal'),
     [PriorEdit(('b', 'intercept'), prior('Normal', [0., 100.]))],
     [('b_0', 'Normal', {'loc': 0., 'scale': 100.}),
      ('b_1', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    ('y ~ 1 + x1 + x2',
     [],
     getfamily('Normal'),
     [PriorEdit(('b', 'x1'), prior('Normal', [0., 100.]))],
     [('b_0', 'Cauchy', {}),
      ('b_1', 'Normal', {'loc': 0., 'scale': 100.}),
      ('b_2', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    # Prior on coef of a factor.
    ('y ~ 1 + x',
     [Categorical('x', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('b', 'x[b]'), prior('Normal', [0., 100.]))],
     [('b_0', 'Cauchy', {}),
      ('b_1', 'Normal', {'loc': 0., 'scale': 100.}),
      ('sigma', 'HalfCauchy', {})]),

    # Prior on coef of an interaction.
    ('y ~ x1:x2',
     [Categorical('x1', list('ab')), Categorical('x2', list('cd'))],
     getfamily('Normal'),
     [PriorEdit(('b', 'x1[b]:x2[c]'), prior('Normal', [0., 100.]))],
     [('b_0', 'Cauchy', {}),
      ('b_1', 'Normal', {'loc': 0., 'scale': 100.}),
      ('b_2', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {})]),

    # Prior on group level `sd` choice.
    ('y ~ 1 + x2 + x3 | x1',
     [Categorical('x1', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('sd', 'x1', 'intercept'), prior('HalfCauchy', [4.]))],
     [('sigma', 'HalfCauchy', {}),
      ('sd_0_0', 'HalfCauchy', {'scale': 4.}),
      ('sd_0_1', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('L_0', 'LKJ', {})]),

    ('y ~ 1 + x2 + x3 || x1',
     [Categorical('x1', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('sd', 'x1', 'intercept'), prior('HalfNormal', [4.]))],
     [('sigma', 'HalfCauchy', {}),
      ('sd_0_0', 'HalfNormal', {'scale': 4.}),
      ('sd_0_1', 'HalfCauchy', {}),
      ('z_0', 'Normal', {})]),

    # Prior on L.
    ('y ~ 1 + x2 | x1',
     [Categorical('x1', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('cor',), prior('LKJ', [2.]))],
     [('sigma', 'HalfCauchy', {}),
      ('sd_0_0', 'HalfCauchy', {}),
      ('z_0', 'Normal', {}),
      ('L_0', 'LKJ', {'eta': 2.})]),

    # Prior on parameter of response distribution.
    ('y ~ x',
     [],
     getfamily('Normal'),
     [PriorEdit(('resp', 'sigma'), prior('HalfCauchy', [4.]))],
     [('b_0', 'Cauchy', {}),
      ('sigma', 'HalfCauchy', {'scale': 4.})]),

    # Custom response family.
    ('y ~ x',
     [],
     apply(getfamily('Normal'), sigma=0.5),
     [],
     [('b_0', 'Cauchy', {})]),

    ('y ~ x',
     [Categorical('y', list('AB'))],
     getfamily('Bernoulli'),
     [],
     [('b_0', 'Cauchy', {})]),

    ('y ~ x',
     [Integral('y', min=0, max=1)],
     getfamily('Bernoulli'),
     [],
     [('b_0', 'Cauchy', {})]),

    ('y ~ x',
     [Integral('y', min=0, max=10)],
     apply(getfamily('Binomial'), num_trials=10),
     [],
     [('b_0', 'Cauchy', {})]),
]

# TODO: Extend this. Could check that the response is observed?
@pytest.mark.parametrize('formula_str, metadata, family, prior_edits, expected', codegen_cases)
def test_pyro_codegen(formula_str, metadata, family, prior_edits, expected):
    # Make dummy data.
    N = 5
    formula = parse(formula_str)
    metadata = build_metadata(formula, metadata)
    df = dummy_df(metadata, N)

    # Define model.
    model = defm(formula_str, df, family, prior_edits)

    # Generate model function and data.
    modelfn = pyro_backend.gen(model.desc).fn
    data = pyro_backend.from_numpy(model.data)

    # Check sample sites.
    trace = poutine.trace(modelfn).get_trace(**data)
    expected_sites = [site for (site, _, _) in expected]
    assert set(trace.stochastic_nodes) - {'obs'} == set(expected_sites)
    for (site, family_name, maybe_params) in expected:
        pyro_family_name = dict(LKJ='LKJCorrCholesky').get(family_name, family_name)
        fn = unwrapfn(trace.nodes[site]['fn'])
        params = maybe_params or default_params[family_name]
        assert type(fn).__name__ == pyro_family_name
        for (name, expected_val) in params.items():
            val = fn.__getattribute__(name)
            assert_equal(val, torch.tensor(expected_val).expand(val.shape))

def unwrapfn(fn):
    return unwrapfn(fn.base_dist) if type(fn) == Independent else fn

@pytest.mark.parametrize('formula_str, metadata, family, prior_edits, expected', codegen_cases)
def test_numpyro_codegen(formula_str, metadata, family, prior_edits, expected):
    # Make dummy data.
    N = 5
    formula = parse(formula_str)
    metadata = build_metadata(formula, metadata)
    df = dummy_df(metadata, N)

    # Define model.
    model = defm(formula_str, df, family, prior_edits)

    # Generate model function and data.
    modelfn = numpyro_backend.gen(model.desc).fn
    data = numpyro_backend.from_numpy(model.data)

    # Check sample sites.
    rng = random.PRNGKey(0)
    trace = numpyro.trace(numpyro.seed(modelfn, rng)).get_trace(**data)
    expected_sites = [site for (site, _, _) in expected]
    sample_sites = [name for name, node in trace.items() if not node['is_observed']]
    assert set(sample_sites) == set(expected_sites)
    for (site, family_name, maybe_params) in expected:
        numpyro_family_name = dict(LKJ='LKJCholesky').get(family_name, family_name)
        fn = trace[site]['fn']
        params = maybe_params or default_params[family_name]
        assert type(fn).__name__ == numpyro_family_name
        for (name, expected_val) in params.items():
            if family_name == 'LKJ':
                assert name == 'eta'
                name = 'concentration'
            val = fn.__getattribute__(name)
            assert_equal(val._value, np.broadcast_to(expected_val, val.shape))

@pytest.mark.parametrize('formula_str, metadata, family, prior_edits, expected', codegen_cases)
@pytest.mark.parametrize('backend', [
    pyro_backend,
    # Set environment variable `RUN_SLOW=1` to run against the NumPyro
    # back end.
    pytest.param(numpyro_backend,
                 marks=pytest.mark.skipif(not os.environ.get('RUN_SLOW', ''), reason='slow'))
])
def test_parameter_shapes(formula_str, metadata, family, prior_edits, expected, backend):
    # Make dummy data.
    N = 5
    formula = parse(formula_str)
    metadata = build_metadata(formula, metadata)
    df = dummy_df(metadata, N)

    # Define model, and generate a single posterior sample.
    model = defm(formula_str, df, family, prior_edits)
    fit = model.fit(backend=backend, iter=1, warmup=0)
    sample = fit.posterior.samples[0]

    # Check parameter sizes.
    for parameter in parameters(model.desc):
        p = fit.posterior.get_param(sample, parameter.name)
        shape = p.shape
        expected_shape = parameter.shape
        #print(shape, expected_shape, parameter, p)
        assert shape == expected_shape
    assert fit.posterior.get_param(sample, 'mu').shape == (N,)


@pytest.mark.parametrize('formula_str, metadata, family, prior_edits', [
    ('y ~ x', [], getfamily('Bernoulli'), []),
    ('y ~ x', [Integral('y', min=0, max=2)], getfamily('Bernoulli'), []),
    ('y ~ x', [Categorical('y', list('abc'))], getfamily('Bernoulli'), []),
    ('y ~ x', [Categorical('y', list('ab'))], getfamily('Normal'), []),
    ('y ~ x', [Integral('y', min=0, max=1)], getfamily('Normal'), []),
    ('y ~ x', [], apply(getfamily('Binomial'), num_trials=1), []),
    ('y ~ x', [Integral('y', min=-1, max=1)], apply(getfamily('Binomial'), num_trials=1), []),
    ('y ~ x',
     [Integral('y', min=0, max=3)],
     apply(getfamily('Binomial'), num_trials=2),
     []),
    ('y ~ x', [Categorical('y', list('abc'))], apply(getfamily('Binomial'), num_trials=1), []),
])
def test_family_and_response_type_checks(formula_str, metadata, family, prior_edits):
    formula = parse(formula_str)
    metadata = build_metadata(formula, metadata)
    design_metadata = designmatrices_metadata(formula, metadata)
    prior_tree = build_prior_tree(formula, design_metadata, family, prior_edits)
    with pytest.raises(Exception, match='not compatible'):
        model = build_model(formula, prior_tree, family, metadata)


@pytest.mark.parametrize('formula_str, metadata, family, prior_edits, expected_error', [
    ('y ~ x',
     [],
     getfamily('Normal'),
     [PriorEdit(('resp', 'sigma'), prior('Normal', [0., 1.]))],
     r'(?i)invalid prior'),
    ('y ~ x1 | x2',
     [Categorical('x2', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('sd', 'x2'), prior('Normal', [0., 1.]))],
     r'(?i)invalid prior'),
    ('y ~ 1 + x1 | x2',
     [Categorical('x2', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('cor', 'x2'), prior('Normal', [0., 1.]))],
     r'(?i)invalid prior'),
    ('y ~ x',
     [],
     getfamily('Normal'),
     [PriorEdit(('b',), prior('Bernoulli', [.5]))],
     r'(?i)invalid prior'),
    ('y ~ x',
     [Integral('y', 0, 1)],
     getfamily('Binomial'),
     [],
     r'(?i)prior missing'),
])
def test_prior_checks(formula_str, metadata, family, prior_edits, expected_error):
    formula = parse(formula_str)
    metadata = build_metadata(formula, metadata)
    design_metadata = designmatrices_metadata(formula, metadata)
    with pytest.raises(Exception, match=expected_error):
        build_prior_tree(formula, design_metadata, family, prior_edits)

@pytest.mark.parametrize('formula_str, df, expected', [
    # (Formula('y', [], []),
    #  pd.DataFrame(dict(y=[1, 2, 3])),
    #  dict(X=torch.tensor([[],
    #                       [],
    #                       []]),
    #       y_obs=torch.tensor([1., 2., 3.]))),
    ('y ~ 1',
     pd.DataFrame(dict(y=[1., 2., 3.])),
     dict(X=np.array([[1.],
                          [1.],
                          [1.]]),
          y_obs=np.array([1., 2., 3.]))),
    ('y ~ x',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=[4., 5., 6.])),
     dict(X=np.array([[4.],
                          [5.],
                          [6.]]),
          y_obs=np.array([1., 2., 3.]))),
    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=[4., 5., 6.])),
     dict(X=np.array([[1., 4.],
                          [1., 5.],
                          [1., 6.]]),
          y_obs=np.array([1., 2., 3.]))),
    ('y ~ x + 1',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=[4., 5., 6.])),
     dict(X=np.array([[1., 4.],
                          [1., 5.],
                          [1., 6.]]),
          y_obs=np.array([1., 2., 3.]))),

    ('y ~ x',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=pd.Categorical(list('AAB')))),
     dict(X=np.array([[1., 0.],
                          [1., 0.],
                          [0., 1.]]),
          y_obs=np.array([1., 2., 3.]))),
    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=pd.Categorical(list('AAB')))),
     dict(X=np.array([[1., 0.],
                          [1., 0.],
                          [1., 1.]]),
          y_obs=np.array([1., 2., 3.]))),
    ('y ~ x1 + x2',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x1=pd.Categorical(list('AAB')),
                       x2=pd.Categorical(list('ABC')))),
     dict(X=np.array([[1., 0., 0., 0.],
                          [1., 0., 1., 0.],
                          [0., 1., 0., 1.]]),
          y_obs=np.array([1., 2., 3.]))),

    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1., 2., 3.],
                       x=pd.Categorical(list('ABC')))),
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
     dict(X=np.array([[1.],
                          [1.],
                          [1.]]),
          y_obs=np.array([1., 2., 3.]),
          J_0=np.array([0, 1, 2]),
          Z_0=np.array([[1., 0.],
                            [1., 0.],
                            [1., 1.]]))),

    # Interactions
    # --------------------------------------------------
    ('y ~ x1:x2',
     pd.DataFrame(dict(y=[1., 2., 3., 4.],
                       x1=pd.Categorical(list('ABAB')),
                       x2=pd.Categorical(list('CCDD')))),
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
     #                     1   B   D   BD
     dict(X=np.array([[1., 0., 0., 0.],
                          [1., 1., 0., 0.],
                          [1., 0., 1., 0.],
                          [1., 1., 1., 1.]]),
          y_obs=np.array([1., 2., 3., 4.]))),

    # Integer-valued Factors
    # --------------------------------------------------
    ('y ~ x1 + x2',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x1=[4, 5, 6],
                       x2=[7., 8., 9.])),
     dict(X=np.array([[4., 7.],
                          [5., 8.],
                          [6., 9.]]),
          y_obs=np.array([1., 2., 3.]))),

    # Categorical Response
    # --------------------------------------------------
    ('y ~ x',
     pd.DataFrame(dict(y=pd.Categorical(list('AAB')),
                       x=[1., 2., 3.])),
     dict(X=np.array([[1.],
                          [2.],
                          [3.]]),
          y_obs=np.array([0., 0., 1.]))),

])
def test_designmatrix(formula_str, df, expected):
    data = makedata(parse(formula_str), df)
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

@pytest.mark.parametrize('formula_str, expected_coding', [
    ('y ~ 1', [tuple()]),
    ('y ~ x', [(CodedFactor('x', False),)]),
    ('y ~ 1 + x', [tuple(), (CodedFactor('x', True),)]),
    ('y ~ a:b', [
        (CodedFactor('a', False), CodedFactor('b', False)) # a:b
    ]),
    ('y ~ 1 + a:b', [
        tuple(),                                          # Intercept
        (CodedFactor('b', True),),                        # b-
        (CodedFactor('a', True), CodedFactor('b', False)) # a-:b
    ]),
    ('y ~ 1 + a + a:b', [
        tuple(),                                          # Intercept
        (CodedFactor('a', True),),                        # a-
        (CodedFactor('a', False), CodedFactor('b', True)) # a:b-
    ]),
    ('y ~ 1 + b + a:b', [
        tuple(),                                          # Intercept
        (CodedFactor('b', True),),                        # b-
        (CodedFactor('a', True), CodedFactor('b', False)) # a-:b
    ]),
    ('y ~ 1 + a + b + a:b', [
        tuple(),                                          # Intercept
        (CodedFactor('a', True),),                        # a-
        (CodedFactor('b', True),),                        # b-
        (CodedFactor('a', True), CodedFactor('b', True))  # a-:b-
    ]),
    ('y ~ a:b + a:b:c', [
        (CodedFactor('a', False), CodedFactor('b', False)),                         # a:b
        (CodedFactor('a', False), CodedFactor('b', False), CodedFactor('c', True)), # a:b:c-
    ]),
])
def test_coding(formula_str, expected_coding):
    formula = parse(formula_str)
    assert categorical_coding(formula.terms) == expected_coding

# I expect these to also pass with PYRO_TENSOR_TYPE='torch.FloatTensor'.

def test_marginals_fitted_smoke():
    N = 10
    S = 4
    df = dummy_df(make_metadata_lookup([RealValued('y'),
                                        RealValued('x'),
                                        Categorical('a', list('ab'))]),
                  N)
    fit = brm('y ~ 1 + x + (1 | a)', df, iter=S, warmup=0)
    def chk(arr, expected_shape):
        assert np.all(np.isfinite(arr))
        assert arr.shape == expected_shape
    chk(marginals(fit).array, (6, 7)) # num coefs x num stats
    chk(fitted(fit), (S, N))
    chk(fitted(fit, 'linear'), (S, N))
    chk(fitted(fit, 'response'), (S, N))
