import pandas as pd

from pyro.contrib.brm.formula import parse, Formula
from pyro.contrib.brm.design import makedata, Metadata, metadata_from_df, designmatrices_metadata
from pyro.contrib.brm.fit import Fit
from pyro.contrib.brm.backend import Backend
from pyro.contrib.brm.family import getfamily, Family
from pyro.contrib.brm.priors import build_prior_tree
from pyro.contrib.brm.model import build_model, model_repr
from pyro.contrib.brm.pyro_backend import backend as pyro_backend
from pyro.contrib.brm.backend import data_from_numpy

def makecode(formula, df, family, prior_edits, backend=pyro_backend):
    desc = makedesc(formula, metadata_from_df(df), family, prior_edits)
    return backend.gen(desc).code

def makedesc(formula, metadata, family, prior_edits):
    assert type(formula) == Formula
    assert type(metadata) == Metadata
    assert type(family) == Family
    assert type(prior_edits) == list
    design_metadata = designmatrices_metadata(formula, metadata)
    prior_tree = build_prior_tree(formula, design_metadata, family, prior_edits)
    return build_model(formula, prior_tree, family, metadata)

def defm(formula_str, df, family=None, prior_edits=None):
    assert type(formula_str) == str
    assert type(df) == pd.DataFrame
    assert family is None or type(family) == Family
    assert prior_edits is None or type(prior_edits) == list
    family = family or getfamily('Normal')
    prior_edits = prior_edits or []
    formula = parse(formula_str)
    # TODO: Both `makedata` and `designmatrices_metadata` call
    # `coding` (from design.py) internally. Instead we ought to call
    # this once and share the result. (Perhaps by having the process
    # of generating design matrices always return the metadata, while
    # retaining the ability to generate the metadata without a
    # concrete dataset.)
    #
    # Related: Perhaps design matrices ought to always have metadata
    # (i.e. column names) associated with them, as in Patsy. (This
    desc = makedesc(formula, metadata_from_df(df), family, prior_edits)
    data = makedata(formula, df)
    return DefmResult(formula, desc, data)

# A wrapper around a pair of model and data. Has a friendly `repr` and
# makes it easy to fit the model.
class DefmResult:
    def __init__(self, formula, desc, data):
        self.formula = formula
        self.desc = desc
        self.data = data

    # TODO: I'm not entirely satisfied with this interface. In
    # particular, I don't like that the args. that this takes can
    # depend on the backend / algo used. One alternative is something like:

    # model.fit(backend).nuts(...)
    # model.fit(backend).svi(...)

    # etc.

    # By having separate methods for each inference algo. each can
    # have its own doc string, which seems useful. This interface also
    # makes it possible to get at the generated code without running
    # inference via something like:

    # model.fit(backend).code

    # Perhaps the `fit` method above could be renamed `fitwith` (or
    # `generate`), and the existing `fit` could wrap that if folk
    # think it's useful?

    # I'm not convinced we can't do better still, but this is worth
    # considering.

    def fit(self, backend=pyro_backend, algo='nuts', **kwargs):
        assert type(backend) == Backend
        assert algo in ['nuts', 'svi']
        model = backend.gen(self.desc)
        data = data_from_numpy(backend, self.data)
        posterior = getattr(backend, algo)(data, model, **kwargs)
        return Fit(self.formula, data, self.desc, model, posterior, backend)

    def __repr__(self):
        return model_repr(self.desc)

def brm(formula_str, df, family=None, prior_edits=None, **kwargs):
    return defm(formula_str, df, family, prior_edits).fit(pyro_backend, **kwargs)
