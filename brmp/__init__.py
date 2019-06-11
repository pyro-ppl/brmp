from pyro.infer.mcmc import MCMC, NUTS

from pyro.contrib.brm.formula import parse
from pyro.contrib.brm.codegen import genmodel, eval_model
from pyro.contrib.brm.design import makedata, dfmetadata, make_metadata_lookup, designmatrices_metadata
from pyro.contrib.brm.fit import Fit, pyro_posterior
from pyro.contrib.brm.family import getfamily
from pyro.contrib.brm.priors import build_prior_tree
from pyro.contrib.brm.model import build_model

def makecode(formula, df, family, prior_edits):
    return genmodel(makedesc(formula, df, family, prior_edits))

def makedesc(formula, df, family, prior_edits):
    df_metadata_lu = make_metadata_lookup(dfmetadata(df))
    design_metadata = designmatrices_metadata(formula, df_metadata_lu)
    prior_tree = build_prior_tree(formula, design_metadata, family, prior_edits)
    return build_model(formula, prior_tree, family, df_metadata_lu)

def brm(formula_str, df, family=getfamily('Normal'), prior_edits=[]):
    formula = parse(formula_str)
    model_desc = makedesc(formula, df, family, prior_edits)
    code = genmodel(model_desc)
    model = eval_model(code)
    # TODO: Both `makedata` and `designmatrices_metadata` call
    # `coding` (from design.py) interally. Instead we ought to call
    # this once and share the result. (Perhaps by having the process
    # of generating design matrices always return the metadata, while
    # retaining the ability to generate the metadata without a
    # concrete dataset.)
    data = makedata(formula, df)
    nuts_kernel = NUTS(model, jit_compile=False, adapt_step_size=True)
    run = MCMC(nuts_kernel, num_samples=500, warmup_steps=100).run(**data)
    return Fit(run, code, data, model_desc, pyro_posterior(run))
