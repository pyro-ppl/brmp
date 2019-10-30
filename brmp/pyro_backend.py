import time
from functools import partial
from sys import stderr
from contextlib import contextmanager

import numpy as np
import torch

import pyro
import pyro.poutine as poutine
from brmp.backend import Backend, Model
from brmp.fit import Samples
from brmp.pyro_codegen import gen
from brmp.utils import flatten, unflatten
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
from pyro.optim import Adam


# Available as `contextmanager.nullcontext` in Python 3.7
@contextmanager
def nullcontext(enter_result=None):
    yield enter_result


# A wrapper around the Pyro seed handler, providing handling for the
# case where seed=None.
def seed_ctx_mgr(seed):
    return nullcontext() if seed is None else poutine.handlers.seed(rng_seed=seed)


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


def get_param(samples, name, preserve_chains):
    # Reminder to use correct interface.
    assert not name == 'mu', 'Use `location` to fetch `mu`.'
    # We insert a dummy "chain" dim here as demanded by the
    # `get_param` interface. (We imagine that prior sampling & SVI
    # always produce a single chain.)
    param = get_node_or_return_value(samples, name)
    return param.unsqueeze(0) if preserve_chains else param


def location(modelfn, samples, data):
    # Re-run the model, taking values from the given samples at
    # `sample` sites, and using the given data to compute `mu`.

    def f(trace):
        return poutine.replay(modelfn, trace)(mode='prior_and_mu', **data)['mu']

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
        # `torch.from_numpy()` throws an error, so make a new empty
        # array instead.
        if arr.dtype == np.float64:
            # I expect that when `arr` holds floats they will always
            # be 64 bit. (See `col2numpy` in design.py.)
            out = torch.empty(arr.shape)
            assert out.dtype == default_dtype
            return out
        elif arr.dtype == np.int64:
            out = torch.empty(arr.shape).long()
            assert out.dtype == torch.int64
            return out
        else:
            raise Exception('unsupported array type')
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
    num_chains, num_samples = next(iter(samples.values())).shape[0:2]
    assert all(arr.shape[0:2] == (num_chains, num_samples) for arr in samples.values())

    # Flatten the sample for easier iteration.
    flat_samples = {k: flatten(arr) for k, arr in samples.items()}

    def run(i):
        sample = {k: arr[i] for k, arr in flat_samples.items()}
        return poutine.condition(modelfn, sample)(**data)

    return_values = [run(i) for i in range(num_chains * num_samples)]

    # TODO: It would probably be better to allocate output arrays and
    # fill them as we run the model. However, I'm holding off on
    # making this change since this is good enough, and it's possible
    # this whole approach may be replaced by something vectorized.

    # We know the model structure is static, so names don't change
    # across executions.
    names = return_values[0].keys()
    # Build output dict., restoring chain dim.
    return {name: unflatten(torch.stack([retval[name] for retval in return_values]), num_chains, num_samples)
            for name in names}


def nuts(data, model, iter, warmup, num_chains, seed):
    assert type(data) == dict
    assert type(model) == Model
    assert type(iter) == int
    assert type(warmup) == int
    assert type(num_chains) == int
    assert seed is None or type(seed) == int

    with seed_ctx_mgr(seed):
        nuts_kernel = NUTS(model.fn, jit_compile=False, adapt_step_size=True)
        mcmc = MCMC(nuts_kernel, num_samples=iter, warmup_steps=warmup, num_chains=num_chains)
        mcmc.run(**data)

    samples = mcmc.get_samples(group_by_chain=True)
    transformed_samples = run_model_on_samples_and_data(model.fn, samples, data)

    def loc(d):
        # Optimization: For the data used for inference, values for
        # `mu` are already computed and available from
        # `transformed_samples`.

        # The `location` method on `Samples` isn't expected to
        # preserve the chain dim so we flatten here.
        if d == data:
            return flatten(transformed_samples['mu'])
        else:
            # TODO: This computes more than is necessary. (i.e. It
            # build additional tensors we immediately throw away.)
            # This is minor, but might be worth addressing eventually.
            return flatten(run_model_on_samples_and_data(model.fn, samples, d)['mu'])

    all_samples = dict(samples, **transformed_samples)

    def get_param(name, preserve_chains):
        param = all_samples[name]
        return param if preserve_chains else flatten(param)

    return Samples(all_samples, get_param, loc)


# Ideally we'd simply use `arr[subsample]` to select out a mini batch,
# but doing so is problematic when the design matrix is empty. (More
# detail below.) This function exists to work around the problem.
#
# Even though the following works:
#
# torch.mv(torch.empty(5, 0), torch.empty(0))
# => torch.tensor([0., 0., 0., 0., 0.])
#
# Attempting to index into the (degenerate) matrix breaks things:
#
# torch.mv(torch.empty(5,0)[[0,1]], torch.empty(0))
# => RuntimeError: invalid argument 6
#
def get_mini_batch(arr, subsample):
    dim = arr.dim()
    assert dim == 1 or dim == 2
    if dim == 2 and arr.shape[1] == 0:
        return torch.empty(len(subsample), 0)
    else:
        return arr[subsample]


def svi(data, model, iter, num_samples, seed, autoguide=None, optim=None, subsample_size=None):
    assert type(data) == dict
    assert type(model) == Model
    assert type(iter) == int
    assert type(num_samples) == int
    assert seed is None or type(seed) == int
    assert autoguide is None or callable(autoguide)

    N = next(data.values().__iter__()).shape[0]
    assert all(arr.shape[0] == N for arr in data.values())
    assert (subsample_size is None or
            type(subsample_size) == int and 0 < subsample_size < N)

    # TODO: Fix that this interface doesn't work for
    # `AutoLaplaceApproximation`, which requires different functions
    # to be used for optimisation / collecting samples.
    autoguide = AutoMultivariateNormal if autoguide is None else autoguide
    optim = Adam({'lr': 1e-3}) if optim is None else optim

    guide = autoguide(model.fn)
    svi = SVI(model.fn, guide, optim, loss=Trace_ELBO())
    pyro.clear_param_store()

    t0 = time.time()
    max_iter_str_width = len(str(iter))
    max_out_len = 0

    with seed_ctx_mgr(seed):

        for i in range(iter):
            if subsample_size is None:
                dfN = None
                subsample = None
                data_for_step = data
            else:
                dfN = N
                subsample = torch.randint(0, N, (subsample_size,)).long()
                data_for_step = {k: get_mini_batch(arr, subsample) for k, arr in data.items()}
            loss = svi.step(dfN=dfN, subsample=subsample, **data_for_step)
            t1 = time.time()
            if t1 - t0 > 0.5 or (i + 1) == iter:
                iter_str = str(i + 1).rjust(max_iter_str_width)
                out = 'iter: {} | loss: {:.3f}'.format(iter_str, loss)
                max_out_len = max(max_out_len, len(out))
                # Sending the ANSI code to clear the line doesn't seem to
                # work in notebooks, so instead we pad the output with
                # enough spaces to ensure we overwrite all previous input.
                print('\r{}'.format(out.ljust(max_out_len)), end='', file=stderr)
                t0 = t1
        print(file=stderr)

        # We run the guide to generate traces from the (approx.)
        # posterior. We also run the model against those traces in order
        # to compute transformed parameters, such as `b`, etc.
        def get_model_trace():
            guide_tr = poutine.trace(guide).get_trace()
            model_tr = poutine.trace(poutine.replay(model.fn, trace=guide_tr)).get_trace(mode='prior_only', **data)
            return model_tr

        # Represent the posterior as a bunch of samples, ignoring the
        # possibility that we might plausibly be able to figure out e.g.
        # posterior maginals from the variational parameters.
        samples = [get_model_trace() for _ in range(num_samples)]

    # Unlike the NUTS case, we don't eagerly compute `mu` (for the
    # data set used for inference) when building `Samples#raw_samples`.
    # (This is because it's possible that N is very large since we
    # support subsampling.) Therefore `loc` always computes `mu` from
    # the data and the samples here.
    def loc(d):
        return location(model.fn, samples, d)

    return Samples(samples, partial(get_param, samples), loc)


def prior(data, model, num_samples, seed):
    assert seed is None or type(seed) == int

    def get_model_trace():
        return poutine.trace(model.fn).get_trace(mode='prior_only', **data)

    with seed_ctx_mgr(seed):
        samples = [get_model_trace() for _ in range(num_samples)]

    def loc(d):
        return location(model.fn, samples, d)

    return Samples(samples, partial(get_param, samples), loc)


# This particular back end implements this by generating additional
# code but other approaches are possible.
def sample_response(model, seed, *args):
    assert type(model) == Model
    assert seed is None or type(seed) is int
    with seed_ctx_mgr(seed):
        return model.sample_response_fn(*args)


def expected_response(model, *args):
    assert type(model) == Model
    return model.expected_response_fn(*args)


def inv_link(model, mu):
    assert type(model) == Model
    return model.inv_link_fn(mu)


backend = Backend('Pyro', gen, prior, nuts, svi, sample_response, expected_response, inv_link, from_numpy, to_numpy)
