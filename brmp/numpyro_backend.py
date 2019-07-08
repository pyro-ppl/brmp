from jax import random, vmap
from jax.config import config; config.update("jax_platform_name", "cpu")

from numpyro.handlers import substitute
from numpyro.hmc_util import initialize_model
from numpyro.mcmc import mcmc

from pyro.contrib.brm.backend import Backend, Model, apply_default_hmc_args
from pyro.contrib.brm.fit import Posterior
from pyro.contrib.brm.numpyro_codegen import gen

def get_param(sample, name):
    assert type(sample) == dict
    return sample[name]

# Values returned by `get_param` are already numpy arrays. (When we
# transpose `allSamples` below we index into the underlying numpy
# array. An even if we didn't, we'd still be turning JAX flavoured
# numpy arrays.)
def to_numpy(param):
    return param

# Data is initially represented as numpy arrays, so there's nothing to
# do before we can use that with NumPyro models.
def from_numpy(data):
    return data

def infer(data, model, seed=0, iter=None, warmup=None):
    assert type(data) == dict
    assert type(model) == Model

    iter, warmup = apply_default_hmc_args(iter, warmup)

    rng = random.PRNGKey(seed)
    init_params, potential_fn, constrain_fn = initialize_model(rng, model.fn, **data)
    samples = mcmc(warmup, iter, init_params,
                   potential_fn=potential_fn,
                   constrain_fn=constrain_fn,
                   print_summary=False)

    # In the Pyro back end, each sample is a trace which includes the
    # model's return value. Here, each sample only gives us access to
    # the values generated at `sample` statements. So to get at the
    # values of derived parameters, the strategy here is to re-run the
    # model, substituting sampled values at sample statements, and
    # collecting the values from the return value.

    # TODO: Is there a way to grab these during sampling to avoid
    # recomputing them here?

    # Q: Will vectorizing become impractical when the number of
    # samples gets large?

    dsamples = vmap(lambda sample: substitute(model.fn, sample)(**data))(samples)
    allsamples = dict(samples, **dsamples)

    # `allsamples` is a dict mapping parameter names to numpy arrays
    # of samples. brmp currently expects to be able to iterate over
    # individual samples, so for now we'll transponse the samples we
    # get from NumPyro.

    # TODO: Adjust the back end interface so that we can avoid this
    # transposition? (Don't assume samples are iterable and follow the
    # consequences.)

    names = allsamples.keys()
    for name in names:
        assert allsamples[name].shape[0] == iter

    # TODO: Indexing into the JAX array here is significantly slower
    # than accessing the underlying array (using `_value`) and
    # indexing into that. (Why?) We do this conditionally, as some
    # parameters will (sometimes) be regular numpy rather than JAX
    # arrays. (e.g. `b` when the model does not include any population
    # level coefficients.)
    def unwrap(arr):
        return arr._value if hasattr(arr, '_value') else arr

    allsamplesT = [{name: unwrap(allsamples[name])[i] for name in names}
                   for i in range(iter)]

    return Posterior(allsamplesT, get_param, to_numpy)

# TODO: Make it possible to run inference on a gpu.

backend = Backend('NumPyro', gen, infer, from_numpy)
