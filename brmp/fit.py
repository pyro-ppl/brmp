from collections import namedtuple

import numpy as np

from pyro.contrib.brm.model import model_repr, parameter_names
from pyro.contrib.brm.family import free_param_names

#from pyro.contrib.brm.utils import join

Fit = namedtuple('Fit', 'data model_desc model posterior backend')
Posterior = namedtuple('Posterior', ['samples', 'get_param', 'to_numpy'])

default_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

def format_quantiles(qs):
    return ['{:g}%'.format(q * 100) for q in qs]

def marginal(fit, extractor):
    assert type(fit) == Fit
    samples = fit.posterior.samples
    to_numpy = fit.posterior.to_numpy
    # `extractor` is a function that extracts from a sample some value
    # of interest, which is expected to be in the back end specific
    # representation. These values are mapped to numpy arrays using
    # the function back specific `to_numpy()` function. Once converted
    # to numpy, the extracted values are reshaped into vectors in
    # order that they can be stacked in a single matrix. (This is
    # necessary for e.g. group level `r_i` parameters which are
    # themselves matrices.)
    #
    # Creating this intermediate list is a bit unpleasant -- could
    # fill a pre-allocated array instead.
    #
    return np.stack([to_numpy(extractor(s)).reshape(-1) for s in samples])

def param_marginal(fit, parameter_name):
    get_param = fit.posterior.get_param
    return marginal(fit, lambda s: get_param(s, parameter_name))

# Computes statistics for an array produced by `marginal`.
def marginal_stats(arr, qs):
    assert len(arr.shape) == 2
    assert type(qs) == list
    assert all(0 <= q <= 1 for q in qs)
    mean = np.mean(arr, 0)
    sd = np.std(arr, 0)
    quantiles = np.quantile(arr, qs, 0)
    stacked = np.hstack((mean.reshape((-1, 1)), sd.reshape((-1, 1)), quantiles.T))
    return stacked

# TODO: Would it be better to replace these tables with pandas data
# frames? They also let you get at the underlying data as a numpy
# array (I assume), and have their own pretty printing.
class ArrReprWrapper:
    def __init__(self, array, row_labels, col_labels):
        assert len(array.shape) == 2
        assert row_labels is None or array.shape[0] == len(row_labels)
        assert col_labels is None or array.shape[1] == len(col_labels)
        self.array = array
        self.col_labels = col_labels
        self.row_labels = row_labels

    def __repr__(self):
        # Format a float. 2 decimal places, space for sign.
        def ff(x):
            return '{: .2f}'.format(x)
        table = [[ff(c) for c in r] for r in self.array.tolist()]
        return layout_table(add_labels(table, self.col_labels, self.row_labels))

def add_labels(table, col_labels, row_labels):
    assert type(table) == list
    assert all(type(row) == list for row in table)
    out = [col_labels] if col_labels is not None else []
    out += table
    if row_labels is not None:
        rlabels = row_labels if col_labels is None else [''] + row_labels
        assert len(out) == len(rlabels)
        out = [[name] + r for r, name in zip(out, rlabels)]
    return out

def layout_table(rows):
    out = []
    num_rows = len(rows)
    assert num_rows > 0
    num_cols = len(rows[0])
    assert all(len(row) == num_cols for row in rows)
    max_widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            max_widths[i] = max(max_widths[i], len(cell))
    fmt = ' '.join('{{:>{}}}'.format(mw) for mw in max_widths)
    return '\n'.join(fmt.format(*row) for row in rows)

# brms                                | brmp
# -------------------------------------------------------------
# fitted(fit)                         | fitted(fit)
# fitted(fit, summary=TRUE)           | summary(fitted(fir))
# fitted(dpar='mu', scale='linear')   | fitted(fit, 'linear')
# fitted(dpar='mu', scale='response') | fitted(fit, 'response')

def fitted(fit, what='expectation'):
    assert type(fit) == Fit
    assert what in ['expectation', 'linear', 'response']

    get_param         = fit.posterior.get_param
    expected_response = fit.model.expected_response_fn
    inv_link          = fit.model.inv_link_fn

    def expectation(sample):
        # Fetch the value of each response parameter from the sample.
        args = [get_param(sample, name)
                for name in free_param_names(fit.model_desc.response.family)]
        # Compute the expected value of the response. This is in the
        # representation used by the current back end.
        return expected_response(*args)
    def linear(sample):
        return get_param(sample, 'mu')
    def response(sample):
        return inv_link(get_param(sample, 'mu'))

    f=dict(expectation=expectation, linear=linear, response=response)[what]

    return marginal(fit, f)

# TODO: We could follow brms and make this available via a `summary`
# flag on `fitted`?
def summary(arr, qs=default_quantiles, row_labels=None):
    col_labels = ['mean', 'sd'] + format_quantiles(qs)
    return ArrReprWrapper(marginal_stats(arr, qs), row_labels, col_labels)

def marginals(fit, qs=default_quantiles):
    arrs = []
    row_labels = []
    col_labels = ['mean', 'sd'] + format_quantiles(qs)
    def param_stats(name):
        return marginal_stats(param_marginal(fit, name), qs)
    # Population coefs.
    arrs.append(param_stats('b'))
    row_labels.extend('b_{}'.format(coef) for coef in fit.model_desc.population.coefs)
    # Groups.
    for ix, group in enumerate(fit.model_desc.groups):
        arrs.append(param_stats('sd_{}'.format(ix)))
        row_labels.extend('sd_{}__{}'.format(group.factor.name, coef)
                          for coef in group.coefs)
        arrs.append(param_stats('r_{}'.format(ix)))
        row_labels.extend('r_{}[{},{}]'.format(group.factor.name, level, coef)
                          for level in group.factor.levels
                          for coef in group.coefs)
    # Response parameters.
    for param in fit.model_desc.response.nonlocparams:
        arrs.append(param_stats(param.name))
        row_labels.append(param.name)
    return ArrReprWrapper(np.vstack(arrs), row_labels, col_labels)

def print_model(fit):
    print(model_repr(fit.model_desc))
