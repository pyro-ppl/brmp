import time
from functools import partial

import numpy as np
import torch

from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC

import pyro
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

from pyro.contrib.brm.backend import Backend, Model, apply_default_hmc_args
from pyro.contrib.brm.fit import Posterior
from pyro.contrib.brm.pyro_codegen import gen

def posterior(run):
    return Posterior(run.exec_traces, get_param)

def get_node_or_return_value(samples, name):
    def getp(sample):
        if name in sample.nodes:
            return sample.nodes[name]['value']
        else:
            return sample.nodes['_RETURN']['value'][name]

    # `detach` is only necessary for SVI.

    # Creating this intermediate list is a bit unpleasant -- could
    # fill a pre-allocated array instead.
    #
    return torch.stack([getp(sample).detach() for sample in samples])

def get_param(samples, name):
    # Reminder to use correct interface.
    assert not name == 'mu', 'Use `location` to fetch `mu`.'
    return get_node_or_return_value(samples, name)

def location(modelfn, samples, data):

    # In general, we need to re-run the model, taking values from the
    # given samples at `sample` sites, and using the given data.

    # The current strategy is to patch up the trace so that we can
    # re-run the model as is. One appealing aspect of this is that it
    # might be easy to adapt to this also sample new values for y.
    # (Rather than having to add something analogous to
    # `expected_response_fn` to code gen.) OTOH, it seems like
    # `expected_response_fn` will stick around, since computing the
    # expectation as part of the model seems wasteful during
    # inference, and perhaps it's easy to adapt this to do sampling
    # with out really adding any new code gen.

    # One alternative would be to add a flag to the model that allows
    # the plate & response distribution to be skipped. (Or
    # alternatively generate two versions of the model if avoiding
    # checking the flag during inference is desirable.)

    # Another option, of course, is to codegen an extra function that
    # directly implements the requuired functionality. This function
    # could be vectorized (i.e. operate on multiple samples in
    # parallel) which is nice.

    def f(sample):
        trace = sample.copy()
        trace.remove_node('y')
        # The length of the vector of indices stored at the plate
        # will likely not match the number of data points in the
        # data. (When applying `location` to "new data".) We could
        # explicitly patch up `trace.node['obs']['value']` but
        # simply removing the node seems to work just as well.
        trace.remove_node('obs')
        return poutine.replay(modelfn, trace)(**data)['mu']

    return torch.stack([f(s).detach() for s in samples])

def to_numpy(param_samples):
    return param_samples.numpy()

# Convert numpy arrays to torch tensors. Arrays of floats use torch's
# default dtype.
def from_numpy(arr):
    assert type(arr) == np.ndarray
    default_dtype = torch.get_default_dtype()
    if arr.size == 0:
        # Attempting to convert an empty array using
        # `torch.from_numpy()` throws an error, so make a new
        # empty array instead. I think this can only happen when
        # `arr` holds floats, which at present will always be 64
        # bit. (See `col2numpy` in design.py.)
        assert arr.dtype == np.float64
        out = torch.empty(arr.shape)
        assert out.dtype == default_dtype
        return out
    else:
        out = torch.from_numpy(arr)
        if torch.is_floating_point(out) and not out.dtype == default_dtype:
            out = out.type(default_dtype)
        return out

# TODO: Ideally this would be vectorized. (i.e. We'd compute the
# model's return value for all samples in parallel.) Pyro's
# `predictive` helper does this by wrapping the model in a `plate`.
# This doesn't work here though because, as written, the model
# includes operations (e.g. `torch.mv`, indexing) that don't
# automatically vectorize. It's likely possible to rewrite the model
# so that such a strategy would work. (e.g. Using Pyro's `vindex` for
# indexing.) Another option is to have the backend generate entirely
# differenent code for both vectorized and non-vectorized variants of
# the model.
def run_model_on_samples_and_data(modelfn, samples, data):
    assert type(samples) == dict
    assert len(samples) > 0
    S = list(samples.values())[0].shape[0]
    assert all(arr.shape[0] == S for _, arr in samples.items())

    def run(i):
        sample = {k: arr[i] for k, arr in samples.items()}
        return poutine.condition(modelfn, sample)(**data)

    return_values = [run(i) for i in range(S)]

    # TODO: It would probably be better to allocate output arrays and
    # fill them as we run the model. However, I'm holding off on
    # making this change since this is good enough, and it's possible
    # this whole approach may be replaced by something vectorized.

    # We know the model structure is static, so names don't change
    # across executions.
    names = return_values[0].keys()
    return {name: torch.stack([retval[name] for retval in return_values])
            for name in names}

def nuts(data, model, iter=None, warmup=None):
    assert type(data) == dict
    assert type(model) == Model

    iter, warmup = apply_default_hmc_args(iter, warmup)

    nuts_kernel = NUTS(model.fn, jit_compile=False, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=iter, warmup_steps=warmup)
    mcmc.run(**data)
    samples = mcmc.get_samples()
    transformed_samples = run_model_on_samples_and_data(model.fn, samples, data)

    def loc(d):
        # Optimization: For the data used for inference, values for
        # `mu` are already computed and available from
        # `transformed_samples`.
        if d == data:
            return transformed_samples['mu']
        else:
            # TODO: This computes more than is necessary. (i.e. It
            # build additional tensors we immediately throw away.)
            # This is minor, but might be worth addressing eventually.
            return run_model_on_samples_and_data(model.fn, samples, d)['mu']

    all_samples = dict(samples, **transformed_samples)

    return Posterior(all_samples, lambda name: all_samples[name], loc)

def svi(data, model, iter=None, num_samples=None, autoguide=None, optim=None):
    assert type(data) == dict
    assert type(model) == Model

    assert iter is None or type(iter) == int
    assert num_samples is None or type(num_samples) == int
    assert autoguide is None or callable(autoguide)

    # TODO: Fix that this interface doesn't work for
    # `AutoLaplaceApproximation`, which requires different functions
    # to be used for optimisation / collecting samples.
    autoguide = AutoDiagonalNormal if autoguide is None else autoguide
    optim = Adam({'lr': 1e-3}) if optim is None else optim

    guide = autoguide(model.fn)
    svi = SVI(model.fn, guide, optim, loss=Trace_ELBO())
    pyro.clear_param_store()

    t0 = time.time()
    max_iter_str_width = len(str(iter))
    max_out_len = 0
    for i in range(iter):
        loss = svi.step(**data)
        t1 = time.time()
        if t1 - t0 > 0.5 or (i+1) == iter:
            iter_str = str(i+1).rjust(max_iter_str_width)
            out = 'iter: {} | loss: {:.3f}'.format(iter_str, loss)
            max_out_len = max(max_out_len, len(out))
            # Sending the ANSI code to clear the line doesn't seem to
            # work in notebooks, so instead we pad the output with
            # enough spaces to ensure we overwrite all previous input.
            print('\r{}'.format(out.ljust(max_out_len)), end='')
            t0 = t1
    print()

    # We run the guide to generate traces from the (approx.)
    # posterior. We also run the model against those traces in order
    # to compute transformed parameters, such as `b`, `mu`, etc.
    def get_model_trace():
        guide_tr = poutine.trace(guide).get_trace()
        model_tr = poutine.trace(poutine.replay(model.fn, trace=guide_tr)).get_trace(**data)
        return model_tr

    # Represent the posterior as a bunch of samples, ignoring the
    # possibility that we might plausibly be able to figure out e.g.
    # posterior maginals from the variational parameters.
    samples = [get_model_trace() for _ in range(num_samples)]

    def loc(d):
        # Optimization: For the data used for inference, values for
        # `mu` are already computed and available from the
        # traces/samples.
        if d == data:
            return get_node_or_return_value(samples, 'mu')
        else:
            return location(model.fn, samples, d)

    return Posterior(samples, partial(get_param, samples), loc)

backend = Backend('Pyro', gen, nuts, svi, from_numpy, to_numpy)
