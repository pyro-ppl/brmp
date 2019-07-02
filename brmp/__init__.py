from pyro.contrib.brm.formula import parse
from pyro.contrib.brm.design import makedata, dfmetadata, make_metadata_lookup, designmatrices_metadata
from pyro.contrib.brm.fit import Fit
from pyro.contrib.brm.backend import Backend
from pyro.contrib.brm.family import getfamily
from pyro.contrib.brm.priors import build_prior_tree
from pyro.contrib.brm.model import build_model, model_repr
from pyro.contrib.brm.pyro_backend import backend as pyro_backend

def makecode(formula, df, family, prior_edits, backend=pyro_backend):
    return backend.gen(makedesc(formula, df, family, prior_edits)).code

def makedesc(formula, df, family, prior_edits):
    df_metadata_lu = make_metadata_lookup(dfmetadata(df))
    design_metadata = designmatrices_metadata(formula, df_metadata_lu)
    prior_tree = build_prior_tree(formula, design_metadata, family, prior_edits)
    return build_model(formula, prior_tree, family, df_metadata_lu)

def defm(formula_str, df, family=None, prior_edits=None):
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
    desc = makedesc(formula, df, family, prior_edits)
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
        posterior = backend.infer(self.data, model, **kwargs)
        return Fit(self.data, self.desc, model, posterior, backend)

    def __repr__(self):
        return model_repr(self.desc)

def brm(formula_str, df, family=None, prior_edits=None, iter=None, warmup=None):
    return defm(formula_str, df, family, prior_edits).fit(pyro_backend, iter=iter, warmup=warmup)
