from functools import partial

import numpy as np

from jax import random, vmap
from jax.config import config; config.update("jax_platform_name", "cpu")

import numpyro.handlers as handler
from numpyro.mcmc import MCMC, NUTS

from pyro.contrib.brm.backend import Backend, Model, apply_default_hmc_args
from pyro.contrib.brm.fit import Samples
from pyro.contrib.brm.numpyro_codegen import gen

# The types described in the comments in pyro_backend.py as follows
# in this back end:
#
# bs: dict from parameter names to JAX numpy arrays
# ps: JAX numpy array

def get_param(samples, name):
    assert type(samples) == dict
    # Reminder to use correct interface.
    assert not name == 'mu', 'Use `location` to fetch `mu`.'
    return samples[name]

# Extract the underlying numpy array (rather than using JAX numpy) to
# match the interface exactly.
def to_numpy(param_samples):
    return param_samples._value if hasattr(param_samples, '_value') else param_samples

# I would ideally like to transform the numpy array into a JAX array
# here, in order to comply with the interface as closely as possible.
# However, since I don't know a super cheap way to do this, and given
# that these two types are mostly interchangable, I'll just use the
# identity here.
def from_numpy(data):
    return data

# TODO: Better name.
def run_model_on_samples_and_data(modelfn, samples, data):
    return vmap(lambda sample: handler.substitute(modelfn, sample)(**data, mode='prior_and_mu'))(samples)

def location(original_data, samples, transformed_samples, model_fn, new_data):
    # Optimization: For the data used for inference, values for `mu`
    # are already computed and available from `transformed_samples`.
    if new_data == original_data:
        return transformed_samples['mu']
    else:
        return run_model_on_samples_and_data(model_fn, samples, new_data)['mu']

def nuts(data, model, seed=None, iter=None, warmup=None, num_chains=None):
    assert type(data) == dict
    assert type(model) == Model
    assert seed is None or type(seed) == int

    iter, warmup, num_chains = apply_default_hmc_args(iter, warmup, num_chains)

    if seed is None:
        seed = np.random.randint(0, 2**32, dtype=np.uint32).astype(np.int32)
    rng = random.PRNGKey(seed)

    kernel = NUTS(model.fn)
    # TODO: We could use a way of avoid requiring users to set
    # `--xla_force_host_platform_device_count` manually when
    # `num_chains` > 1 to achieve parallel chains.
    mcmc = MCMC(kernel, warmup, iter, num_chains=num_chains)
    mcmc.run(rng, **data)
    samples = mcmc.get_samples()

    # Here we re-run the model on the samples in order to collect
    # transformed parameters. (e.g. `b`, `mu`, etc.) Theses are made
    # available via the return value of the model.
    transformed_samples = run_model_on_samples_and_data(model.fn, samples, data)
    all_samples = dict(samples, **transformed_samples)

    loc = partial(location, data, samples, transformed_samples, model.fn)

    return Samples(all_samples, partial(get_param, all_samples), loc)

def svi(*args, **kwargs):
    raise NotImplementedError

def prior(data, model, num_samples, seed=None):
    assert type(data) == dict
    assert type(model) == Model
    assert type(num_samples) == int and num_samples > 0
    assert seed is None or type(seed) == int

    if seed is None:
        seed = np.random.randint(0, 2**32, dtype=np.uint32).astype(np.int32)
    rngs = random.split(random.PRNGKey(seed), num_samples)

    def get_model_trace(rng):
        fn = handler.seed(model.fn, rng)
        model_tr = handler.trace(fn).get_trace(mode="prior_only", **data)
        # Unpack the bits of the trace we're interested in into a dict
        # in order to support vectorization. (dicts support
        # vectorization, OrderedDicts, as used by the trace, don't.)
        return {k: node['value'] for k,node in model_tr.items()}

    samples = vmap(get_model_trace)(rngs)
    transformed_samples = run_model_on_samples_and_data(model.fn, samples, data)
    all_samples = dict(samples, **transformed_samples)

    loc = partial(location, data, samples, transformed_samples, model.fn)

    return Samples(all_samples, partial(get_param, all_samples), loc)


# TODO: Make it possible to run inference on a gpu.

backend = Backend('NumPyro', gen, prior, nuts, svi, from_numpy, to_numpy)
