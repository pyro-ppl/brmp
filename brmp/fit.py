from collections import namedtuple
from functools import partial

import numpy as np
import numpyro.diagnostics as diags
import pandas as pd

from brmp.backend import data_from_numpy
from brmp.design import predictors
from brmp.family import free_param_names
from brmp.model import scalar_parameter_map, scalar_parameter_names
from brmp.utils import flatten

default_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# `Fit` carries around `formula`, `metadata` and `contrasts` for the
# sole purpose of being able to encode any new data passed to
# `fitted`.

# TODO: Consider storing `formula`, `metadata` and `contrasts` on
# `ModelDescPre` (and then `ModelDesc`) as an alternative to storing
# them on `Fit`. (Since it seems more natural.)

# One snag is that `ModelDescPre` only sees the lengths of any custom
# coding and not the full matrix. However, while deferring having to
# give a concrete data frame is useful (because it saves having to
# make up data to tinker with a model), it's not clear that deferring
# having to give contrasts has a similar benefit.


class Fit(namedtuple('Fit', 'formula metadata contrasts data model_desc assets samples backend')):

    # TODO: This doesn't match the brms interface, but the deviation
    # aren't improvements either. Figure out what to do about that.

    # brms                                               | brmp
    # -----------------------------------------------------------------------------------
    # fitted(fit, summary=FALSE)                         | fit.fitted()
    # fitted(dpar='mu', scale='linear', summary=FALSE)   | fit.fitted('linear')
    # fitted(dpar='mu', scale='response', summary=FALSE) | fit.fitted('response')
    # fitted(fit, newdata=..., summary=FALSE)            | fit.fitted(data=...)
    # fitted(fit, ..., summary=TRUE)                     | summary(fit.fitted(...))
    # predict(fit, summary=FALSE)                        | fit.fitted('sample')
    # predict(fit, summary=TRUE)                         | summary(fit.fitted('sample'))

    # https://rdrr.io/cran/brms/man/fitted.brmsfit.html

    def __init__(self, *args, **kwargs):
        self._parameter_map = scalar_parameter_map(self.model_desc)

    def fitted(self, what='expectation', data=None, seed=None):
        """
        Produces predictions from the fitted model.

        Predicted values are computed for each sample collected during inference,
        and for each row in the data set.

        :param what: The value to predict. Valid arguments and their effect are described below:

                     .. list-table::
                        :widths: auto

                        * - ``'expectation'``
                          - Computes the expected value of the response distribution.
                        * - ``'sample'``
                          - Draws a sample from the response distribution.
                        * - ``'response'``
                          - Computes the output of the model followed by any
                            inverse link function. i.e. The value of the location
                            parameter of the response distribution.
                        * - ``'linear'``
                          - Computes the output of the model prior to the application
                            of any inverse link function.

        :type what: str
        :param data: The data from which to compute the predictions. When omitted,
                     the data on which the model was fit is used.


        :type data: pandas.DataFrame
        :param seed: Random seed. Used only when ``'sample'`` is given as the ``'what'`` argument.
        :type seed: int
        :return: An array with shape ``(S, N)``. Where ``S`` is the number of samples taken
                 during inference and ``N`` is the number of rows in the data set used for prediction.
        :rtype: numpy.ndarray

        """
        assert what in ['sample', 'expectation', 'linear', 'response']
        assert data is None or type(data) is pd.DataFrame
        assert seed is None or type(seed) == int

        get_param = self.samples.get_param
        location = self.samples.location
        to_numpy = self.backend.to_numpy
        expected_response = partial(self.backend.expected_response, self.assets)
        sample_response = partial(self.backend.sample_response, self.assets, seed)
        inv_link = partial(self.backend.inv_link, self.assets)

        mu = location(self.data if data is None
                      else data_from_numpy(self.backend, predictors(self.formula, data, self.metadata, self.contrasts)))

        if what == 'sample' or what == 'expectation':
            args = [mu if name == 'mu' else get_param(name, False)
                    for name in free_param_names(self.model_desc.response.family)]
            response_fn = sample_response if what == 'sample' else expected_response
            return to_numpy(response_fn(*args))
        elif what == 'linear':
            return to_numpy(mu)
        elif what == 'response':
            return to_numpy(inv_link(mu))
        else:
            raise ValueError('Unhandled value of the `what` parameter encountered.')

    # Similar to the following:
    # https://rdrr.io/cran/rstan/man/stanfit-method-summary.html

    # TODO: This produces the same output as the old implementation of
    # `marginal`, though it's less efficient. Can the previous efficiency
    # be recovered? The problem is that we pull out each individual scalar
    # parameter as a vector and then stack those, rather than just stack
    # entire parameters as before. One thought is that such an
    # optimisation might be best pushed into `get_scalar_param`. i.e. This
    # might accept a list of a parameter names and return the
    # corresponding scalar parameters stacked into a matrix. The aim would
    # be to do this without performing any unnecessary slicing. (Though
    # this sounds fiddly.)
    def marginals(self, qs=default_quantiles):
        """Produces a table containing statistics of the marginal
        distibutions of the parameters of the fitted model.

        :param qs: A list of quantiles to include in the output.
        :type qs: list
        :return: A table of marginal statistics.
        :rtype: brmp.fit.ArrReprWrapper

        Example::

          fit = brm('y ~ x', df).fit()
          print(fit.marginals())

          #        mean    sd  2.5%   25%   50%   75% 97.5% n_eff r_hat
          # b_x    0.42  0.33 -0.11  0.14  0.48  0.65  0.88  5.18  1.00
          # sigma  0.78  0.28  0.48  0.61  0.68  0.87  1.32  5.28  1.10

        """
        names = scalar_parameter_names(self.model_desc)
        vecs = [self.get_scalar_param(name, True) for name in names]
        col_labels = ['mean', 'sd'] + format_quantiles(qs) + ['n_eff', 'r_hat']
        samples = np.stack(vecs, axis=2)
        stats_arr = marginal_stats(flatten(samples), qs)
        n_eff = compute_diag_or_default(effective_sample_size, samples)
        r_hat = compute_diag_or_default(gelman_rubin, samples)
        arr = np.hstack([stats_arr, n_eff[..., np.newaxis], r_hat[..., np.newaxis]])
        return ArrReprWrapper(arr, names, col_labels)

    # A back end agnostic wrapper around back end specific implementations
    # of `fit.samples.get_param`.
    def get_param(self, name, preserve_chains=False):
        return self.backend.to_numpy(self.samples.get_param(name, preserve_chains))

    # TODO: If parameter and scalar parameter names never clash, perhaps
    # having a single lookup method would be convenient. Perhaps this
    # could be wired up to `fit.samples[...]`?

    # TODO: Mention other ways of obtaining valid parameter names?

    def get_scalar_param(self, name, preserve_chains=False):
        """
        Extracts the values sampled for a single parameter from a model fit.

        :param name: The name of a parameter of the model. Valid names are those
                     shown in the output of :func:`~brmp.fit.marginals`.
        :type name: str
        :param preserve_chains: Whether to group samples by the MCMC chain on which
                                they were collected.
        :type preserve_chains: bool
        :return: An array with shape ``(S,)`` when ``preserve_chains=False``, ``(C, S)``
                 otherwise. Where ``S`` is the number of samples taken during inference,
                 and ``C`` is the number of MCMC chains run.

        :rtype: numpy.ndarray

        """
        res = [p for (n, p) in self._parameter_map if n == name]
        assert len(res) < 2
        if len(res) == 0:
            raise KeyError('unknown parameter name: {}'.format(name))
        param_name, index = res[0]
        # Construct a slice to pick out the given index at all chains (if
        # present) and all samples.
        slc = (Ellipsis,) + index
        return self.get_param(param_name, preserve_chains)[slc]

    # TODO: Consider delegating to `marginals` or similar?
    def __repr__(self):
        # The repr of namedtuple ends up long and not very useful for
        # Fit. This is similar to the default implementation of repr
        # used for classes.
        return '<brmp.fit.Fit at {}>'.format(hex(id(self)))


Samples = namedtuple('Samples', ['raw_samples', 'get_param', 'location'])


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


# TODO: We could follow brms and make this available via a `summary`
# flag on `fitted`?
def summary(arr, qs=default_quantiles, row_labels=None):
    col_labels = ['mean', 'sd'] + format_quantiles(qs)
    return ArrReprWrapper(marginal_stats(arr, qs), row_labels, col_labels)


def gelman_rubin(samples):
    if ((samples.shape[0] < 2 and samples.shape[1] < 4) or
            (samples.shape[0] >= 2 and samples.shape[1] < 2)):
        return None  # Too few chains or samples.
    elif samples.shape[0] >= 2:
        return diags.gelman_rubin(samples)
    else:
        return diags.split_gelman_rubin(samples)


def effective_sample_size(samples):
    if samples.shape[1] < 2:
        return None  # Too few samples.
    else:
        return diags.effective_sample_size(samples)


def compute_diag_or_default(diag, samples):
    val = diag(samples)
    if val is not None:
        return val
    else:
        return np.full((samples.shape[2],), np.nan)
