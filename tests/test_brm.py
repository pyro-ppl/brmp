import pytest

import torch
import pandas as pd

import pyro.poutine as poutine
from pyro.distributions import Independent, Normal, Cauchy, HalfCauchy, LKJCorrCholesky

from pyro.contrib.brm.formula import parse
from pyro.contrib.brm.codegen import genmodel, eval_model
from pyro.contrib.brm.design import dummydata, Factor, makedata, make_metadata_lookup, designmatrices_metadata
from pyro.contrib.brm.priors import prior, Prior, PriorEdit, get_response_prior
from pyro.contrib.brm.family import getfamily, FAMILIES

from tests.common import assert_equal

default_params = dict(
    Normal          = dict(loc=0., scale=1.),
    Cauchy          = dict(loc=0., scale=1.),
    HalfCauchy      = dict(scale=3.),
    LKJCorrCholesky = dict(eta=1.),
)

# TODO: Extend this. Could check shapes of sampled values? (Although
# there are already asserting in the generated code to do that.) Check
# response is observed.
@pytest.mark.parametrize('formula_str, metadata, family, prior_edits, expected', [
    # TODO: This (and similar examples below) can't be expressed with
    # the current parser. Is it useful to fix this (`y ~ -1`?), or can
    # these be dropped?
    #(Formula('y', [], []), [], [], ['sigma']),

    ('y ~ 1 + x', [], getfamily('Normal'), [],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {})]),

    ('y ~ 1 + x1 + x2', [], getfamily('Normal'), [],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {})]),

    #(Formula('y', [], [Group([], 'z', True)]), [Factor('z', list('ab'))], [], ['sigma', 'z_1']),
    # Groups with fewer than two terms don't sample the (Cholesky
    # decomp. of the) correlation matrix.
    #(Formula('y', [], [Group([], 'z', True)]), [Factor('z', list('ab'))], [], ['sigma', 'z_1']),
    ('y ~ 1 | z', [Factor('z', list('ab'))], getfamily('Normal'), [],
     [('sigma', HalfCauchy, {}),
      ('z_1', Normal, {}),
      ('sd_1_0', HalfCauchy, {})]),

    ('y ~ x | z', [Factor('z', list('ab'))], getfamily('Normal'), [],
     [('sigma', HalfCauchy, {}),
      ('z_1', Normal, {}),
      ('sd_1_0', HalfCauchy, {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 | z)', [Factor('z', list('ab'))], getfamily('Normal'), [],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {}),
      ('z_1', Normal, {}),
      ('sd_1_0', HalfCauchy, {}),
      ('L_1', LKJCorrCholesky, {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 || z)', [Factor('z', list('ab'))], getfamily('Normal'), [],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {}),
      ('z_1', Normal, {}),
      ('sd_1_0', HalfCauchy, {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 + x4 | z1) + (1 + x5 | z2)',
     [Factor('z1', list('ab')), Factor('z2', list('ab'))],
     getfamily('Normal'),
     [],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {}),
      ('z_1', Normal, {}),
      ('sd_1_0', HalfCauchy, {}),
      ('L_1', LKJCorrCholesky, {}),
      ('z_2', Normal, {}),
      ('sd_2_0', HalfCauchy, {}),
      ('L_2', LKJCorrCholesky, {})]),

    # Custom priors.
    ('y ~ 1 + x1 + x2',
     [],
     getfamily('Normal'),
     [PriorEdit(('b',), prior('Normal', [0., 100.]))],
     [('b_0', Normal, {'loc': 0., 'scale': 100.}),
      ('sigma', HalfCauchy, {})]),

    ('y ~ 1 + x1 + x2',
     [],
     getfamily('Normal'),
     [PriorEdit(('b', 'intercept'), prior('Normal', [0., 100.]))],
     [('b_0', Normal, {'loc': 0., 'scale': 100.}),
      ('b_1', Cauchy, {}),
      ('sigma', HalfCauchy, {})]),

    ('y ~ 1 + x1 + x2',
     [],
     getfamily('Normal'),
     [PriorEdit(('b', 'x1'), prior('Normal', [0., 100.]))],
     [('b_0', Cauchy, {}),
      ('b_1', Normal, {'loc': 0., 'scale': 100.}),
      ('b_2', Cauchy, {}),
      ('sigma', HalfCauchy, {})]),

    # Prior on coef of a factor.
    ('y ~ 1 + x',
     [Factor('x', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('b', 'x[b]'), prior('Normal', [0., 100.]))],
     [('b_0', Cauchy, {}),
      ('b_1', Normal, {'loc': 0., 'scale': 100.}),
      ('sigma', HalfCauchy, {})]),

    # Prior on group level `sd` choice.
    ('y ~ 1 + x2 + x3 | x1',
     [Factor('x1', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('sd', 'x1', 'intercept'), prior('HalfCauchy', [4.]))],
     [('sigma', HalfCauchy, {}),
      ('sd_1_0', HalfCauchy, {'scale': 4.}),
      ('sd_1_1', HalfCauchy, {}),
      ('z_1', Normal, {}),
      ('L_1', LKJCorrCholesky, {})]),

    ('y ~ 1 + x2 + x3 || x1',
     [Factor('x1', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('sd', 'x1', 'intercept'), prior('HalfCauchy', [4.]))],
     [('sigma', HalfCauchy, {}),
      ('sd_1_0', HalfCauchy, {'scale': 4.}),
      ('sd_1_1', HalfCauchy, {}),
      ('z_1', Normal, {})]),

    # Prior on L.
    ('y ~ 1 + x2 | x1',
     [Factor('x1', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('cor',), prior('LKJ', [2.]))],
     [('sigma', HalfCauchy, {}),
      ('sd_1_0', HalfCauchy, {}),
      ('z_1', Normal, {}),
      ('L_1', LKJCorrCholesky, {'eta': 2.})]),

    # Prior on parameter of response distribution.
    ('y ~ x',
     [],
     getfamily('Normal'),
     [PriorEdit(('resp', 'sigma'), prior('HalfCauchy', [4.]))],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {'scale': 4.})]),

    # Custom response family.
    ('y ~ x',
     [],
     getfamily('Bernoulli'),
     [],
     [('b_0', Cauchy, {})]),

])
def test_codegen(formula_str, metadata, family, prior_edits, expected):
    formula = parse(formula_str)
    metadata = make_metadata_lookup(metadata)
    code = genmodel(formula, metadata, family, prior_edits)
    #print(code)
    model = eval_model(code)
    data = dummydata(formula, metadata, 5)
    trace = poutine.trace(model).get_trace(**data)
    expected_sites = [site for (site, _, _) in expected]
    assert set(trace.stochastic_nodes) - {'obs'} == set(expected_sites)
    for (site, family, maybe_params) in expected:
        fn = unwrapfn(trace.nodes[site]['fn'])
        params = maybe_params or default_params[fn.__class__.__name__]
        assert type(fn) == family
        for (name, expected_val) in params.items():
            val = fn.__getattribute__(name)
            assert_equal(val, torch.tensor(expected_val).expand(val.shape))


def unwrapfn(fn):
    return unwrapfn(fn.base_dist) if type(fn) == Independent else fn

@pytest.mark.parametrize('formula_str, df, expected', [
    # (Formula('y', [], []),
    #  pd.DataFrame(dict(y=[1, 2, 3])),
    #  dict(X=torch.tensor([[],
    #                       [],
    #                       []]),
    #       y_obs=torch.tensor([1., 2., 3.]))),
    ('y ~ 1',
     pd.DataFrame(dict(y=[1, 2, 3])),
     dict(X=torch.tensor([[1.],
                          [1.],
                          [1.]]),
          y_obs=torch.tensor([1., 2., 3.]))),
    ('y ~ x',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=[4, 5, 6])),
     dict(X=torch.tensor([[4.],
                          [5.],
                          [6.]]),
          y_obs=torch.tensor([1., 2., 3.]))),
    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=[4, 5, 6])),
     dict(X=torch.tensor([[1., 4.],
                          [1., 5.],
                          [1., 6.]]),
          y_obs=torch.tensor([1., 2., 3.]))),
    ('y ~ x + 1',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=[4, 5, 6])),
     dict(X=torch.tensor([[1., 4.],
                          [1., 5.],
                          [1., 6.]]),
          y_obs=torch.tensor([1., 2., 3.]))),

    ('y ~ x',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=pd.Categorical(list('AAB')))),
     dict(X=torch.tensor([[1., 0.],
                          [1., 0.],
                          [0., 1.]]),
          y_obs=torch.tensor([1., 2., 3.]))),
    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=pd.Categorical(list('AAB')))),
     dict(X=torch.tensor([[1., 0.],
                          [1., 0.],
                          [1., 1.]]),
          y_obs=torch.tensor([1., 2., 3.]))),
    ('y ~ x1 + x2',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x1=pd.Categorical(list('AAB')),
                       x2=pd.Categorical(list('ABC')))),
     dict(X=torch.tensor([[1., 0., 0., 0.],
                          [1., 0., 1., 0.],
                          [0., 1., 0., 1.]]),
          y_obs=torch.tensor([1., 2., 3.]))),

    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=pd.Categorical(list('ABC')))),
     dict(X=torch.tensor([[1., 0., 0.],
                          [1., 1., 0.],
                          [1., 0., 1.]]),
          y_obs=torch.tensor([1., 2., 3.]))),

    # (Formula('y', [], [Group([], 'x', True)]),
    #  pd.DataFrame(dict(y=[1, 2, 3],
    #                    x=pd.Categorical(list('ABC')))),
    #  dict(X=torch.tensor([[],
    #                       [],
    #                       []]),
    #       y_obs=torch.tensor([1., 2., 3.]),
    #       J_1=torch.tensor([0, 1, 2]),
    #       Z_1=torch.tensor([[],
    #                         [],
    #                         []]))),
    ('y ~ 1 + (1 + x1 | x2)',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x1=pd.Categorical(list('AAB')),
                       x2=pd.Categorical(list('ABC')))),
     dict(X=torch.tensor([[1.],
                          [1.],
                          [1.]]),
          y_obs=torch.tensor([1., 2., 3.]),
          J_1=torch.tensor([0, 1, 2]),
          Z_1=torch.tensor([[1., 0.],
                            [1., 0.],
                            [1., 1.]]))),
])
def test_designmatrix(formula_str, df, expected):
    data = makedata(parse(formula_str), df)
    assert set(data.keys()) == set(expected.keys())
    for k in expected.keys():
        assert_equal(data[k], expected[k])

def test_response_priors_is_complete():
    for family in FAMILIES:
        if family.response is not None:
            for param in family.params:
                if not param == family.response.param:
                    assert type(get_response_prior(family.name, param)) == Prior
