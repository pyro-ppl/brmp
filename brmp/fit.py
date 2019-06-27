from collections import namedtuple

import numpy as np

from pyro.contrib.brm.model import model_repr, parameter_names
from pyro.contrib.brm.family import free_param_names

# TODO: Should invlinkfn & expected_response_fn be on Posterior with
# other back end specific fns?

Fit = namedtuple('Fit', ['run', 'code', 'data', 'model', 'posterior', 'invlinkfn', 'expected_response_fn'])
Posterior = namedtuple('Posterior', ['samples', 'get_param', 'to_numpy'])

# The idea is that `pyro_posterior` and `pyro_get_param` capture the
# backend specific part of processing posterior samples. Alternatives
# to this approach include:

# 1. Have each back end return an iterable of samples, where each
# sample is something like a dictionary holding all of the parameters
# of interest. (Effectively the backend would be returning the result
# of mapping `get_param` over every sample for every parameter.

# 2. Have each backend implement some kind of query interface,
# allowing things like `query.marginal('b').mean()`, etc.

def pyro_posterior(run):
    return Posterior(run.exec_traces, pyro_get_param, pyro_to_numpy)

# Extracts a value of interest (e.g. 'b', 'r_0', 'L_1', 'sigma') from
# a single sample.

# It's expected that this should support all parameter names returned
# by `parameter_names(model)` where `model` is the `Model` from which
# samples were drawn. It should also support fetching the (final)
# value bound to `mu` in the generated code.
def pyro_get_param(sample, name):
    if name in sample.nodes:
        return sample.nodes[name]['value']
    else:
        return sample.nodes['_RETURN']['value'][name]

# This provides a back-end specific method for turning a parameter
# value (as returned by `get_param`) into a numpy array.
def pyro_to_numpy(param):
    return param.numpy()

def marginals(fit, qs):
    params = parameter_names(fit.model)
    return {p: marginal(fit, p, qs) for p in params}

def marginal(fit, parameter_name, qs):
    assert type(fit) == Fit
    assert type(parameter_name) == str
    assert type(qs) == list
    assert all(0 <= q <= 1 for q in qs)
    samples = fit.posterior.samples
    get_param = fit.posterior.get_param
    to_numpy = fit.posterior.to_numpy
    samples_arr = np.stack([to_numpy(get_param(sample, parameter_name))
                            for sample in samples])
    mean = np.mean(samples_arr, 0)
    sd = np.std(samples_arr, 0)
    quantiles = np.quantile(samples_arr, qs, 0)
    return mean, sd, quantiles

# TODO: Extract new function from the bits that are shared between
# this and `marginal`.

# brms                                | brmp
# -------------------------------------------------------------
# fitted(fit)                         | fitted(fit)
# fitted(dpar='mu', scale='linear')   | fitted(fit, 'linear')
# fitted(dpar='mu', scale='response') | fitted(fit, 'response')

def fitted(fit, what='expectation'):
    assert type(fit) == Fit
    assert what in ['expectation', 'linear', 'response']

    samples   = fit.posterior.samples
    get_param = fit.posterior.get_param
    to_numpy  = fit.posterior.to_numpy

    def expectation(sample):
        # Fetch the value of each response parameter from the sample.
        args = [get_param(sample, name)
                for name in free_param_names(fit.model.response.family)]
        # Compute the expected value of the response. This is in the
        # representation used by the current back end.
        return fit.expected_response_fn(*args)
    def linear(sample):
        return get_param(sample, 'mu')
    def response(sample):
        return fit.invlinkfn(get_param(sample, 'mu'))

    f=dict(expectation=expectation, linear=linear, response=response)[what]

    return np.stack([to_numpy(f(s)) for s in samples])

default_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# TODO: Have this return a wrapper around the numpy array. (`stacked`
# in the code below.) The wrapper can have something like print table
# as its `__repr__()` implementation, and an e.g. `array` property to
# access the underlying data. Or, perhaps better yet, have this return
# a data frame? (Likewise for other similar methods?)

# TODO: The tables this produces and those produced by
# `print_marginals` look very similar -- consolidate?

# `summary(fitted(fit))` is intended to be similar to
# `fitted(fit, summary=TRUE)` in brms.

def summary(arr, qs=default_quantiles, labels=None):
    # `arr` is expected to be an array of sample. The first index
    # ranges over samples.
    N = arr.shape[1]
    # TODO: Check is iterable.
    assert labels is None or len(labels) == N
    mean = np.mean(arr, 0)
    sd = np.std(arr, 0)
    quantiles = np.quantile(arr, qs, 0)
    stacked = np.hstack([mean.reshape((-1, 1)), sd.reshape((-1, 1)), quantiles.T])
    header = [['mean', 'sd'] + ['{:g}%'.format(q * 100) for q in qs]]
    rows = header + [[ff(c) for c in r] for r in stacked.tolist()]
    if labels is not None:
        rows = [[label] + row for label, row in zip([''] + list(labels), rows)]
    print_table(rows)

# TODO: Have this be generated by the `__repr__` method on `Fit`?
# Allowing users to `print(fit)` to see this?
def print_marginals_simple(fit):
    for name, (mean, sd, _) in marginals(fit).items():
        print('==================================================')
        print(name)
        print('-- mean ------------------------------------------')
        print(mean)
        print('-- stddev ----------------------------------------')
        print(sd)

# Format a float.
def ff(x):
    return '{: .2f}'.format(x)

# This relies on the assumption that all models make available the
# parameters described by the `parameters` function in model.py, and
# that each of these is a tensor/multi-dimensional array of the
# expected size.
def print_marginals(fit, qs=default_quantiles):
    rows = [['', 'mean', 'sd'] + ['{:g}%'.format(q * 100) for q in qs]]
    mean_and_sd = marginals(fit, qs)
    b_mean, b_sd, b_quantiles = mean_and_sd['b']
    assert len(fit.model.population.coefs) == len(b_mean) == len(b_sd) == len(b_quantiles.T)
    for coef, mean, sd, quantiles in zip(fit.model.population.coefs, b_mean, b_sd, b_quantiles.T):
        readable_name = 'b_{}'.format(coef)
        rows.append([readable_name, ff(mean), ff(sd)] + [ff(q) for q in quantiles])
    for ix, group in enumerate(fit.model.groups):
        r_mean, r_sd, r_quantiles = mean_and_sd['r_{}'.format(ix)]
        for i, level in enumerate(group.factor.levels):
            for j, coef in enumerate(group.coefs):
                readable_name = 'r_{}[{},{}]'.format(group.factor.name, level, coef)
                rows.append([readable_name, ff(r_mean[i, j]), ff(r_sd[i, j])] + [ff(q) for q in r_quantiles[:, i, j]])

    for param in fit.model.response.nonlocparams:
        param_mean, param_sd, param_quantiles = mean_and_sd[param.name]
        rows.append([param.name, ff(param_mean[0]), ff(param_sd[0])] + [ff(q) for q in param_quantiles[:, 0]])

    print_table(rows)

def print_table(rows):
    num_rows = len(rows)
    assert num_rows > 0
    num_cols = len(rows[0])
    assert all(len(row) == num_cols for row in rows)
    max_widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            max_widths[i] = max(max_widths[i], len(cell))
    fmt = ' '.join('{{:>{}}}'.format(mw) for mw in max_widths)
    for row in rows:
        print(fmt.format(*row))

def print_model(fit):
    print(model_repr(fit.model))
