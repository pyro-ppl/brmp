import pytest

import torch
import pandas as pd

import pyro.poutine as poutine

from pyro.contrib.brm.formula import parse
from pyro.contrib.brm.codegen import genmodel, eval_model
from pyro.contrib.brm.design import dummydata, Factor, makedata, make_metadata_lookup, designmatrices_metadata
from pyro.contrib.brm.priors import Prior, PriorEdit

from tests.common import assert_equal

# TODO: Extend this. Could check that each random choice comes from
# expected family? Could check shapes of sampled values? (Although
# there are already asserting in the generated code to do that.) Check
# response is observed.
@pytest.mark.parametrize('formula_str, metadata, prior_edits, expected', [
    # TODO: This (and similar examples below) can't be expressed with
    # the current parser. Is it useful to fix this (`y ~ -1`?), or can
    # these be dropped?
    #(Formula('y', [], []), [], [], ['sigma']),

    ('y ~ 1 + x', [], [], ['b_0', 'sigma']),
    ('y ~ 1 + x1 + x2', [], [], ['b_0', 'sigma']),

    #(Formula('y', [], [Group([], 'z', True)]), [Factor('z', list('ab'))], [], ['sigma', 'z_1']),
    # Groups with fewer than two terms don't sample the (Cholesky
    # decomp. of the) correlation matrix.
    #(Formula('y', [], [Group([], 'z', True)]), [Factor('z', list('ab'))], [], ['sigma', 'z_1']),
    ('y ~ 1 | z', [Factor('z', list('ab'))], [], ['sigma', 'z_1', 'sd_1_0']),
    ('y ~ x | z', [Factor('z', list('ab'))], [], ['sigma', 'z_1', 'sd_1_0']),

    ('y ~ 1 + x1 + x2 + (1 + x3 | z)', [Factor('z', list('ab'))], [], ['b_0', 'sigma', 'z_1', 'sd_1_0', 'L_1']),
    ('y ~ 1 + x1 + x2 + (1 + x3 || z)', [Factor('z', list('ab'))], [], ['b_0', 'sigma', 'z_1', 'sd_1_0']),
    ('y ~ 1 + x1 + x2 + (1 + x3 + x4 | z1) + (1 + x5 | z2)',
     [Factor('z1', list('ab')), Factor('z2', list('ab'))],
     [],
     ['b_0', 'sigma', 'z_1', 'sd_1_0', 'L_1', 'z_2', 'sd_2_0', 'L_2']),

    # Custom priors.
    # TODO: Check that values are sampled from the distribution specified.
    ('y ~ 1 + x1 + x2',
     [],
     [PriorEdit(['b'], Prior('Normal', [0., 100.]))],
     ['b_0', 'sigma']),

    ('y ~ 1 + x1 + x2',
     [],
     [PriorEdit(['b', 'intercept'], Prior('Normal', [0., 100.]))],
     ['b_0', 'b_1', 'sigma']),

    ('y ~ 1 + x1 + x2',
     [],
     [PriorEdit(['b', 'x1'], Prior('Normal', [0., 100.]))],
     ['b_0', 'b_1', 'b_2', 'sigma']),

    # Prior on coef of a factor.
    ('y ~ 1 + x',
     [Factor('x', list('ab'))],
     [PriorEdit(['b', 'x[b]'], Prior('Normal', [0., 100.]))],
     ['b_0', 'b_1', 'sigma']),

    # Prior on group level `sd` choice.
    ('y ~ 1 + x2 + x3 | x1',
     [Factor('x1', list('ab'))],
     [PriorEdit(['sd', 'x1', 'intercept'], Prior('HalfCauchy', [4.]))],
     ['sigma', 'sd_1_0', 'sd_1_1', 'z_1', 'L_1']),

    ('y ~ 1 + x2 + x3 || x1',
     [Factor('x1', list('ab'))],
     [PriorEdit(['sd', 'x1', 'intercept'], Prior('HalfCauchy', [4.]))],
     ['sigma', 'sd_1_0', 'sd_1_1', 'z_1']),

    # Prior on L.
    ('y ~ 1 + x2 | x1',
     [Factor('x1', list('ab'))],
     [PriorEdit(['L'], Prior('LKJ', [2.]))],
     ['sigma', 'sd_1_0', 'z_1', 'L_1']),

])
def test_codegen(formula_str, metadata, prior_edits, expected):
    formula = parse(formula_str)
    metadata = make_metadata_lookup(metadata)
    code = genmodel(formula, metadata, prior_edits)
    #print(code)
    model = eval_model(code)
    data = dummydata(formula, metadata, 5)
    trace = poutine.trace(model).get_trace(**data)
    assert set(trace.stochastic_nodes) - {'obs'} == set(expected)

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
