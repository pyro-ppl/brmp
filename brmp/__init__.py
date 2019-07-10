import pandas as pd

from pyro.contrib.brm.formula import parse, Formula
from pyro.contrib.brm.design import makedata, dfmetadata, make_metadata_lookup, designmatrices_metadata
from pyro.contrib.brm.fit import Fit
from pyro.contrib.brm.backend import Backend
from pyro.contrib.brm.family import getfamily, Family
from pyro.contrib.brm.priors import build_prior_tree
from pyro.contrib.brm.model import build_model, model_repr
from pyro.contrib.brm.pyro_backend import backend as pyro_backend

def makecode(formula, df, family, prior_edits, backend=pyro_backend):
    desc = makedesc(formula, dfmetadata(df), family, prior_edits)
    return backend.gen(desc).code

def makedesc(formula, df_metadata, family, prior_edits):
    assert type(formula) == Formula
    assert type(df_metadata) == list
    assert type(family) == Family
    assert type(prior_edits) == list
    df_metadata_lu = make_metadata_lookup(df_metadata)
    design_metadata = designmatrices_metadata(formula, df_metadata_lu)
    prior_tree = build_prior_tree(formula, design_metadata, family, prior_edits)
    return build_model(formula, prior_tree, family, df_metadata_lu)

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
    desc = makedesc(formula, dfmetadata(df), family, prior_edits)
    data = makedata(formula, df)
    return DefmResult(desc, data)

# A wrapper around a pair of model and data. Has a friendly `repr` and
# makes it easy to fit the model.
class DefmResult:
    def __init__(self, desc, data):
        self.desc = desc
        self.data = data

    def fit(self, backend=pyro_backend, **kwargs):
        assert type(backend) == Backend
        model = backend.gen(self.desc)
        data = {k: backend.from_numpy(arr) for k, arr in self.data.items()}
        posterior = backend.infer(data, model, **kwargs)
        return Fit(data, self.desc, model, posterior, backend)

    def __repr__(self):
        return model_repr(self.desc)

def brm(formula_str, df, family=None, prior_edits=None, **kwargs):
    return defm(formula_str, df, family, prior_edits).fit(pyro_backend, **kwargs)
