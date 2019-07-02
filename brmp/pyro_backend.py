from pyro.infer.mcmc import MCMC, NUTS

from pyro.contrib.brm.backend import Backend, GenModel
from pyro.contrib.brm.fit import Posterior
from pyro.contrib.brm.codegen import gen

# The idea is that `pyro_posterior` and `pyro_get_param` capture the
# backend specific part of processing posterior samples. Alternatives
# to this approach include:

# 1. Have each back end return an iterable of samples, where each
# sample is something like a dictionary holding all of the parameters
# of interest. (Effectively the backend would be returning the result
# of mapping `get_param` over every sample for every parameter.

# 2. Have each backend implement some kind of query interface,
# allowing things like `query.marginal('b').mean()`, etc.

def posterior(run):
    return Posterior(run.exec_traces, get_param, to_numpy)

# Extracts a value of interest (e.g. 'b', 'r_0', 'L_1', 'sigma') from
# a single sample.

# It's expected that this should support all parameter names returned
# by `parameter_names(model)` where `model` is the `ModelDesc` from
# which samples were drawn. It should also support fetching the
# (final) value bound to `mu` in the generated code.
def get_param(sample, name):
    if name in sample.nodes:
        return sample.nodes[name]['value']
    else:
        return sample.nodes['_RETURN']['value'][name]

# This provides a back-end specific method for turning a parameter
# value (as returned by `get_param`) into a numpy array.
def to_numpy(param):
    return param.numpy()

def infer(data, generated_model, iter, warmup):
    assert type(data) == dict
    assert type(generated_model) == GenModel

    # TODO: Turn the data into the format required by this particular
    # backend.

    iter = 10 if iter is None else iter
    warmup = iter // 2 if warmup is None else warmup

    nuts_kernel = NUTS(generated_model.fn, jit_compile=False, adapt_step_size=True)
    run = MCMC(nuts_kernel, num_samples=iter, warmup_steps=warmup).run(**data)

    return posterior(run)

backend = Backend('Pyro', gen, infer)
