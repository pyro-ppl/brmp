from pyro.infer.mcmc import MCMC, NUTS

from pyro.contrib.brm.codegen import genmodel, eval_model
from pyro.contrib.brm.design import makedata, dfmetadata, make_metadata_lookup
from pyro.contrib.brm.fit import Fit

def makecode(formula, df, prior_edits):
    return genmodel(formula, make_metadata_lookup(dfmetadata(df)), prior_edits)

def brm(formula, df, prior_edits=[]):
    code = makecode(formula, df, prior_edits)
    model = eval_model(code)
    data = makedata(formula, df)
    nuts_kernel = NUTS(model, jit_compile=False, adapt_step_size=True)
    run = MCMC(nuts_kernel, num_samples=500, warmup_steps=100).run(**data)
    return Fit(run, code, data)
