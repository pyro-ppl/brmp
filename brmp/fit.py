from collections import namedtuple

import numpy as np

from pyro.contrib.brm.model import model_repr, parameter_names
from pyro.contrib.brm.family import free_param_names

#from pyro.contrib.brm.utils import join

Fit = namedtuple('Fit', 'data model_desc model posterior backend')
Posterior = namedtuple('Posterior', ['samples', 'get_param', 'to_numpy'])

def param_marginal(posterior, parameter_name):
    return posterior.to_numpy(posterior.get_param(posterior.samples, parameter_name))

default_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

def format_quantiles(qs):
    return ['{:g}%'.format(q * 100) for q in qs]

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

# https://rdrr.io/cran/brms/man/fitted.brmsfit.html

def fitted(fit, what='expectation'):
    assert type(fit) == Fit
    assert what in ['expectation', 'linear', 'response']

    samples           = fit.posterior.samples
    get_param         = fit.posterior.get_param
    to_numpy          = fit.posterior.to_numpy
    expected_response = fit.model.expected_response_fn
    inv_link          = fit.model.inv_link_fn

    if what == 'expectation':
        args = [get_param(samples, name)
                for name in free_param_names(fit.model_desc.response.family)]
        return to_numpy(expected_response(*args))
    elif what == 'linear':
        return to_numpy(get_param(samples, 'mu'))
    elif what == 'response':
        return to_numpy(inv_link(get_param(samples, 'mu')))
    else:
        raise 'Unhandled value of the `what` parameter encountered.'


# TODO: We could follow brms and make this available via a `summary`
# flag on `fitted`?
def summary(arr, qs=default_quantiles, row_labels=None):
    col_labels = ['mean', 'sd'] + format_quantiles(qs)
    return ArrReprWrapper(marginal_stats(arr, qs), row_labels, col_labels)

# Similar to the following:
# https://rdrr.io/cran/rstan/man/stanfit-method-summary.html
def marginals(fit, qs=default_quantiles):
    arrs = []
    row_labels = []
    col_labels = ['mean', 'sd'] + format_quantiles(qs)
    def flatten(arr):
        return arr.reshape((arr.shape[0], -1))
    def param_stats(name):
        return marginal_stats(flatten(param_marginal(fit.posterior, name)), qs)
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
