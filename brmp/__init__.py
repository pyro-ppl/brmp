import numpy as np
import pandas as pd

from brmp.formula import parse, Formula
from brmp.design import makedata, Metadata, metadata_from_df, code_lengths
from brmp.fit import Fit
from brmp.backend import Backend
from brmp.family import Family, Normal
from brmp.priors import build_prior_tree
from brmp.model_pre import build_model_pre
from brmp.model import build_model, model_repr
from brmp.pyro_backend import backend as pyro_backend
from brmp.backend import data_from_numpy

_default_backend = pyro_backend

def makedesc(formula, metadata, family, priors, code_lengths):
    assert type(formula) == Formula
    assert type(metadata) == Metadata
    assert type(family) == Family
    assert type(priors) == list
    model_desc_pre = build_model_pre(formula, metadata, family, code_lengths)
    prior_tree = build_prior_tree(model_desc_pre, priors)
    return build_model(model_desc_pre, prior_tree)

def defm(formula_str, df, family=None, priors=None, contrasts=None):
    """
    Defines a model and encodes data in design matrices.

    By default categorical columns are coded using dummy coding.

    :param formula_str: An lme4 formula. e.g. ``'y ~ 1 + x'``. See
                        :class:`~brmp.formula.Formula` for a description
                        of the supported syntax.
    :type formula_str: str
    :param df: A data frame containing columns for each of the variables in
               ``formula_str``.
    :type df: pandas.DataFrame
    :param family: The model's response family.
    :type family: brmp.family.Family
    :param priors: A list of :class:`~brmp.priors.Prior` instances describing the model's priors.
    :type priors: list
    :param contrasts: A dictionary that optionally maps variable names to contrast matrices describing
                      custom encodings of categorical variables. Each contrast matrix should be
                      a :class:`~numpy.ndarray` of shape ``(L, C)``, where ``L`` is the number of levels
                      present in the categorical variable and ``C`` is the length of the desired
                      encoding.
    :type contrasts: dict
    :return: A wrapper around the model description and the design matrices.
    :rtype: brmp.DefmResult

    Example::

      df = pd.DataFrame({'y': [1., 2.], 'x': [.5, 0.]})
      model = defm('y ~ 1 + x', df)


    """
    assert type(formula_str) == str
    assert type(df) == pd.DataFrame
    assert family is None or type(family) == Family
    assert priors is None or type(priors) == list
    assert contrasts is None or type(contrasts) == dict

    family = family or Normal
    priors = priors or []
    contrasts = contrasts or {}

    # TODO: Consider accepting nested arrays as well as numpy arrays.
    # (If we do, convert to numpy arrays here in `defm`?)
    assert all(type(val) == np.ndarray and len(val.shape) == 2 for val in contrasts.values())

    formula = parse(formula_str)
    # Perhaps design matrices ought to always have metadata (i.e.
    # column names) associated with them, as in Patsy.
    metadata = metadata_from_df(df)
    desc = makedesc(formula, metadata, family, priors, code_lengths(contrasts))
    data = makedata(formula, df, metadata, contrasts)
    return DefmResult(formula, metadata, contrasts, desc, data)

# A wrapper around a pair of model and data. Has a friendly `repr` and
# makes it easy to fit the model.
class DefmResult:
    def __init__(self, formula, metadata, contrasts, desc, data):
        self.formula = formula
        self.metadata = metadata
        self.contrasts = contrasts
        self.desc = desc
        # TODO: Turn this into a `@property` to improve generate docs?
        self.data = data
        """
        A dictionary with the encoded data as its values.
        Each value is a :class:`~numpy.ndarray`.
        """

    def fit(self, backend=_default_backend, algo='nuts', **kwargs):
        """
        Fits the wrapped model.

        :param backend: The backend used to perform inference.
        :type backend: brmp.backend.Backend
        :param algo: The algorithm used for inference, either ``'nuts'``, ``'svi'`` or ``'prior'``.
        :type algo: str
        :param kwargs: Inference algorithm-specific keyword arguments. See the methods on
                       :class:`~brmp.GenerateResult` for details.
        :return: A model fit.
        :rtype: brmp.fit.Fit

        Example::

          fit = defm('y ~ x', df).fit()

        """
        assert type(backend) == Backend
        assert algo in ['prior', 'nuts', 'svi']
        return getattr(self.generate(backend), algo)(**kwargs)

    # Generate model code and data from this description, using the
    # supplied backend.
    def generate(self, backend=_default_backend):
        """
        Generates the assets required to fit the wrapped model.

        :param backend: The backend used to generate code and other assets.
        :type backend: brmp.backend.Backend
        :return: A wrapper around the generated assets.
        :rtype: brmp.GenerateResult
        """
        assert type(backend) == Backend
        model = backend.gen(self.desc)
        data = data_from_numpy(backend, self.data)
        return GenerateResult(self, backend, model, data)

    def prior(self, *args, **kwargs):
        return self.generate().prior(*args, **kwargs)

    def nuts(self, *args, **kwargs):
        return self.generate().nuts(*args, **kwargs)

    def svi(self, *args, **kwargs):
        return self.generate().svi(*args, **kwargs)

    def __repr__(self):
        return model_repr(self.desc)

# A wrapper around the result of calling DefmResult#generate. Exists
# to support the following interface:
#
# model.generate(<backend>).nuts(...)
# model.generate(<backend>).svi(...)
#
# This makes it possible to get at the code generated by a backend
# without running inference. For example:
#
# model.generate(<backend>).model.code
#
class GenerateResult():
    def __init__(self, defm_result, backend, model, data):
        self.defm_result = defm_result
        self.backend = backend
        self.model = model
        self.data = data

    def _run_algo(self, algo, *args, **kwargs):
        samples = getattr(self.backend, algo)(self.data, self.model, *args, **kwargs)
        return Fit(self.defm_result.formula, self.defm_result.metadata, self.defm_result.contrasts, self.data, self.defm_result.desc, self.model, samples, self.backend)

    def prior(self, num_samples=10, *args, **kwargs):
        """
        Sample from the prior.

        :param num_samples: The number of samples to take.
        :type num_samples: int
        :return: A model fit.
        :rtype: brmp.fit.Fit

        Example::

          fit = defm('y ~ x', df).generate().prior()

        """
        return self._run_algo('prior', num_samples, *args, **kwargs)

    def nuts(self, iter=10, warmup=None, num_chains=1, *args, **kwargs):
        """
        Fit the model using NUTS.

        :param iter: The number of (post warm up) samples to take.
        :type iter: int
        :param warmup: The number of warm up samples to take. Warm up samples are
                       not included in the final model fit. Defaults to
                       ``iter / 2``.
        :type warmup: int
        :param num_chains: The number of chains to run.
        :type num_chains: int
        :return: A model fit.
        :rtype: brmp.fit.Fit

        Example::

          fit = defm('y ~ x', df).generate().nuts()

        """
        warmup = iter // 2 if warmup is None else warmup
        return self._run_algo('nuts', iter, warmup, num_chains, *args, **kwargs)

    def svi(self, iter=10, num_samples=10, *args, **kwargs):
        """
        Fit the model using stochastic variational inference.

        :param iter: The number of optimisation steps to take.
        :type iter: int
        :param num_samples: The number of samples to take from the variational
                            posterior.
        :type num_samples: int
        :return: A model fit.
        :rtype: brmp.fit.Fit

        Example::

          fit = defm('y ~ x', df).generate().svi()

        """
        return self._run_algo('svi', iter, num_samples, *args, **kwargs)


def brm(formula_str, df, family=None, priors=None, **kwargs):
    return defm(formula_str, df, family, priors).fit(_default_backend, **kwargs)
