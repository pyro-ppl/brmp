from functools import partial

import numpy as np
import numpyro.handlers as handler
from jax import random, vmap
from jax.config import config
from numpyro.infer import MCMC, NUTS

from brmp.backend import Backend, Model
from brmp.fit import Samples
from brmp.numpyro_codegen import gen
from brmp.utils import flatten, unflatten

config.update("jax_platform_name", "cpu")


# The types described in the comments in pyro_backend.py as follows
# in this back end:
#
# bs: dict from parameter names to JAX numpy arrays
# ps: JAX numpy array

def get_param(samples, name, preserve_chains):
    assert type(samples) == dict
    # Reminder to use correct interface.
    assert not name == 'mu', 'Use `location` to fetch `mu`.'
    param = samples[name]
    return param if preserve_chains else flatten(param)


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
    assert type(samples) == dict
    assert len(samples) > 0
    num_chains, num_samples = next(iter(samples.values())).shape[0:2]
    assert all(arr.shape[0:2] == (num_chains, num_samples) for arr in samples.values())
    flat_samples = {k: flatten(arr) for k, arr in samples.items()}
    out = vmap(lambda sample: handler.substitute(modelfn, sample)(**data, mode='prior_and_mu'))(flat_samples)
    # Restore chain dim.
    return {k: unflatten(arr, num_chains, num_samples) for k, arr in out.items()}


def location(original_data, samples, transformed_samples, model_fn, new_data):
    # Optimization: For the data used for inference, values for `mu`
    # are already computed and available from `transformed_samples`.
    if new_data == original_data:
        return flatten(transformed_samples['mu'])
    else:
        return flatten(run_model_on_samples_and_data(model_fn, samples, new_data)['mu'])


def nuts(data, model, iter, warmup, num_chains, seed=None):
    assert type(data) == dict
    assert type(model) == Model
    assert type(iter) == int
    assert type(warmup) == int
    assert type(num_chains) == int
    assert seed is None or type(seed) == int

    if seed is None:
        seed = np.random.randint(0, 2 ** 32, dtype=np.uint32).astype(np.int32)
    rng = random.PRNGKey(seed)

    kernel = NUTS(model.fn)
    # TODO: We could use a way of avoid requiring users to set
    # `--xla_force_host_platform_device_count` manually when
    # `num_chains` > 1 to achieve parallel chains.
    mcmc = MCMC(kernel, warmup, iter, num_chains=num_chains)
    mcmc.run(rng, **data)
    samples = mcmc.get_samples(group_by_chain=True)

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
        seed = np.random.randint(0, 2 ** 32, dtype=np.uint32).astype(np.int32)
    rngs = random.split(random.PRNGKey(seed), num_samples)

    def get_model_trace(rng):
        fn = handler.seed(model.fn, rng)
        model_tr = handler.trace(fn).get_trace(mode="prior_only", **data)
        # Unpack the bits of the trace we're interested in into a dict
        # in order to support vectorization. (dicts support
        # vectorization, OrderedDicts, as used by the trace, don't.)
        return {k: node['value'] for k, node in model_tr.items()}

    flat_samples = vmap(get_model_trace)(rngs)
    # Insert dummy "chain" dim.
    samples = {k: np.expand_dims(arr, 0) for k, arr in flat_samples.items()}
    transformed_samples = run_model_on_samples_and_data(model.fn, samples, data)
    all_samples = dict(samples, **transformed_samples)

    loc = partial(location, data, samples, transformed_samples, model.fn)

    return Samples(all_samples, partial(get_param, all_samples), loc)


# This particular back end implements this by generating additional
# code but other approaches are possible.
def sample_response(model, *args):
    assert type(model) == Model
    return model.sample_response_fn(*args)


def expected_response(model, *args):
    assert type(model) == Model
    return model.expected_response_fn(*args)


def inv_link(model, mu):
    assert type(model) == Model
    return model.inv_link_fn(mu)


# TODO: Make it possible to run inference on a gpu.

backend = Backend('NumPyro', gen, prior, nuts, svi, sample_response, expected_response, inv_link, from_numpy, to_numpy)
