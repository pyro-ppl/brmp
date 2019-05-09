import pytest

import torch
import pandas as pd

import pyro.poutine as poutine

from pyro.contrib.brm.formula import Formula, Group, _1
from pyro.contrib.brm.codegen import genmodel, eval_model
from pyro.contrib.brm.design import dummydata, Factor, makedata, make_metadata_lookup

from tests.common import assert_equal

# TODO: Extend this. Could check that each random choice comes from
# expected family? Could check shapes of sampled values? (Although
# there are already asserting in the generated code to do that.) Check
# response is observed.
@pytest.mark.parametrize('formula, metadata, expected', [
    (Formula('y', [], []), [], ['b', 'sigma']),
    (Formula('y', [_1, 'x'], []), [], ['b', 'sigma']),
    (Formula('y', [_1, 'x1', 'x2'], []), [], ['b', 'sigma']),

    (Formula('y', [], [Group([], 'z', True)]), [Factor('z', list('ab'))], ['b', 'sigma', 'z_1', 'sd_1']),

    # Groups with less than two terms don't sample the (Cholesky
    # decomp. of the) correlation matrix.
    (Formula('y', [], [Group([], 'z', True)]), [Factor('z', list('ab'))], ['b', 'sigma', 'z_1', 'sd_1']),
    (Formula('y', [], [Group([_1], 'z', True)]), [Factor('z', list('ab'))], ['b', 'sigma', 'z_1', 'sd_1']),
    (Formula('y', [], [Group(['x'], 'z', True)]), [Factor('z', list('ab'))], ['b', 'sigma', 'z_1', 'sd_1']),

    (Formula('y', [_1, 'x1', 'x2'], [Group([_1, 'x3'],'z', True)]), [Factor('z', list('ab'))], ['b', 'sigma', 'z_1', 'sd_1', 'L_1']),
    (Formula('y', [_1, 'x1', 'x2'], [Group([_1, 'x3'],'z', False)]), [Factor('z', list('ab'))], ['b', 'sigma', 'z_1', 'sd_1']),
    (Formula('y', [_1, 'x1', 'x2'], [Group([_1, 'x3', 'x4'], 'z1', True), Group([_1, 'x5'], 'z2', True)]),
     [Factor('z1', list('ab')), Factor('z2', list('ab'))],
     ['b', 'sigma', 'z_1', 'sd_1', 'L_1', 'z_2', 'sd_2', 'L_2']),
])
def test_codegen(formula, metadata, expected):
    metadata = make_metadata_lookup(metadata)
    model = eval_model(genmodel(formula, metadata))
    data = dummydata(formula, metadata, 5)
    trace = poutine.trace(model).get_trace(**data)
    assert set(trace.stochastic_nodes) - {'obs'} == set(expected)

@pytest.mark.parametrize('formula, df, expected', [
    (Formula('y', [], []),
     pd.DataFrame(dict(y=[1, 2, 3])),
     dict(X=torch.tensor([[],
                          [],
                          []]),
          y_obs=torch.tensor([1., 2., 3.]))),
    (Formula('y', [_1], []),
     pd.DataFrame(dict(y=[1, 2, 3])),
     dict(X=torch.tensor([[1.],
                          [1.],
                          [1.]]),
          y_obs=torch.tensor([1., 2., 3.]))),
    (Formula('y', ['x'], []),
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=[4, 5, 6])),
     dict(X=torch.tensor([[4.],
                          [5.],
                          [6.]]),
          y_obs=torch.tensor([1., 2., 3.]))),
    (Formula('y', [_1, 'x'], []),
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=[4, 5, 6])),
     dict(X=torch.tensor([[1., 4.],
                          [1., 5.],
                          [1., 6.]]),
          y_obs=torch.tensor([1., 2., 3.]))),
    (Formula('y', ['x', _1], []),
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=[4, 5, 6])),
     dict(X=torch.tensor([[1., 4.],
                          [1., 5.],
                          [1., 6.]]),
          y_obs=torch.tensor([1., 2., 3.]))),

    (Formula('y', ['x'], []),
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=pd.Categorical(list('AAB')))),
     dict(X=torch.tensor([[1., 0.],
                          [1., 0.],
                          [0., 1.]]),
          y_obs=torch.tensor([1., 2., 3.]))),
    (Formula('y', [_1, 'x'], []),
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=pd.Categorical(list('AAB')))),
     dict(X=torch.tensor([[1., 0.],
                          [1., 0.],
                          [1., 1.]]),
          y_obs=torch.tensor([1., 2., 3.]))),
    (Formula('y', ['x1', 'x2'], []),
     pd.DataFrame(dict(y=[1, 2, 3],
                       x1=pd.Categorical(list('AAB')),
                       x2=pd.Categorical(list('ABC')))),
     dict(X=torch.tensor([[1., 0., 0., 0.],
                          [1., 0., 1., 0.],
                          [0., 1., 0., 1.]]),
          y_obs=torch.tensor([1., 2., 3.]))),

    (Formula('y', [_1, 'x'], []),
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=pd.Categorical(list('ABC')))),
     dict(X=torch.tensor([[1., 0., 0.],
                          [1., 1., 0.],
                          [1., 0., 1.]]),
          y_obs=torch.tensor([1., 2., 3.]))),

    (Formula('y', [], [Group([], 'x', True)]),
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=pd.Categorical(list('ABC')))),
     dict(X=torch.tensor([[],
                          [],
                          []]),
          y_obs=torch.tensor([1., 2., 3.]),
          J_1=torch.tensor([0, 1, 2]),
          Z_1=torch.tensor([[],
                            [],
                            []]))),
    (Formula('y', [_1], [Group([_1, 'x1'], 'x2', True)]),
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
def test_designmatrix(formula, df, expected):
    data = makedata(formula, df)
    assert set(data.keys()) == set(expected.keys())
    for k in expected.keys():
        assert_equal(data[k], expected[k])
