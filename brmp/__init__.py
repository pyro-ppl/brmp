from pyro.infer.mcmc import MCMC, NUTS

from pyro.contrib.brm.formula import parse
from pyro.contrib.brm.codegen import genmodel, eval_model
from pyro.contrib.brm.design import makedata, dfmetadata, make_metadata_lookup
from pyro.contrib.brm.fit import Fit
from pyro.contrib.brm.family import getfamily

def makecode(formula, df, family, prior_edits):
    return genmodel(formula, make_metadata_lookup(dfmetadata(df)), family, prior_edits)

def brm(formula_str, df, family=getfamily('Normal'), prior_edits=[]):
    formula = parse(formula_str)
    code = makecode(formula, df, family, prior_edits)
    model = eval_model(code)
    data = makedata(formula, df)
    nuts_kernel = NUTS(model, jit_compile=False, adapt_step_size=True)
    run = MCMC(nuts_kernel, num_samples=500, warmup_steps=100).run(**data)
    return Fit(run, code, data)
