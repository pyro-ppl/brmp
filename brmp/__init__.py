import numpy as np
import pandas as pd

from brmp.backend import Backend, data_from_numpy
from brmp.design import Metadata, code_lengths, makedata, metadata_from_df
from brmp.family import Family, Normal
from brmp.fit import Fit
from brmp.formula import Formula, parse
from brmp.model import build_model, model_repr
from brmp.model_pre import build_model_pre
from brmp.priors import build_prior_tree
from brmp.pyro_backend import backend as pyro_backend
from brmp.numpyro_backend import backend as numpyro_backend


def makedesc(formula, metadata, family, priors, code_lengths):
    assert type(formula) == Formula
    assert type(metadata) == Metadata
    assert type(family) == Family
    assert type(priors) == list
    model_desc_pre = build_model_pre(formula, metadata, family, code_lengths)
    prior_tree = build_prior_tree(model_desc_pre, priors)
    return build_model(model_desc_pre, prior_tree)

# TODO: In principle we only need to pass `code_length(contrasts)`
# here. The actual coding isn't required until we see data.


def define_model(formula_str, metadata, family=None, priors=None, contrasts=None):
    assert type(formula_str) == str
    assert type(metadata) == Metadata
    assert family is None or type(family) == Family
    assert priors is None or type(priors) == list
    assert contrasts is None or type(contrasts) == dict

    family = family or Normal
    priors = priors or []
    contrasts = contrasts or {}

    # TODO: Consider accepting nested arrays as well as numpy arrays.
    # (If we do, convert to numpy arrays here in `define_model`?)
    assert all(type(val) == np.ndarray and len(val.shape) == 2 for val in contrasts.values())

    formula = parse(formula_str)
    desc = makedesc(formula, metadata, family, priors, code_lengths(contrasts))
    return Model(formula, metadata, contrasts, desc)


class Model:
    def __init__(self, formula, metadata, contrasts, desc):
        self.formula = formula
        self.metadata = metadata
        self.contrasts = contrasts
        self.desc = desc

    def gen(self, backend):
        assets = backend.gen(self.desc)
        return AssetsWrapper(self, assets, backend)

    # Generate design matrices. (Represented as numpy arrays.)
    def encode(self, df):
        return makedata(self.formula, df, self.metadata, self.contrasts)


class AssetsWrapper:
    def __init__(self, model, assets, backend):
        assert type(backend) == Backend
        self.model = model
        self.assets = assets
        self.backend = backend

    def encode(self, df):
        data = self.model.encode(df)
        return data_from_numpy(self.backend, data)

    def run_algo(self, name, data, *args, **kwargs):
        samples = getattr(self.backend, name)(data, self.assets, *args, **kwargs)
        return Fit(self.model.formula, self.model.metadata,
                   self.model.contrasts, data,
                   self.model.desc, self.assets, samples, self.backend)


def brm(formula_str, df, family=None, priors=None, contrasts=None):
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
    :rtype: brmp.ModelAndData

    Example::

      df = pd.DataFrame({'y': [1., 2.], 'x': [.5, 0.]})
      model = brm('y ~ 1 + x', df)

    """
    assert type(formula_str) == str
    assert type(df) == pd.DataFrame
    assert family is None or type(family) == Family
    assert priors is None or type(priors) == list
    assert contrasts is None or type(contrasts) == dict
    metadata = metadata_from_df(df)
    model = define_model(formula_str, metadata, family, priors, contrasts)
    data = model.encode(df)
    return ModelAndData(model, df, data)


class ModelAndData:
    def __init__(self, model, df, data):
        self.model = model
        self.df = df
        # TODO: Turn this into a `@property` to improve generate docs?
        self.data = data
        """
        A dictionary with the encoded data as its values.
        Each value is a :class:`~numpy.ndarray`.
        """

        # We might eventually want to cache generated assets. This
        # will likely be particularly useful once it's possible to
        # cache compiled NumPyro models on the `Model` instance.

        # Once we have this, we can extend `nuts` etc. to take an
        # extra (optional) `df` argument, enabling the model to be fit
        # with a data frame other than that used to define the model.
        # This would be passed to `run_algo` (as a keyword argument)
        # which already implements the required logic.

        # One thing we might consider adding with this is checks to
        # ensure that such a data frame is compatible (in an
        # appropriate sense) with the data frame used to define the
        # model.

    def run_algo(self, name, backend, *args, df=None, **kwargs):
        assert type(backend) == Backend
        data = self.model.encode(df) if df is not None else self.data
        assets_wrapper = self.model.gen(backend)
        return assets_wrapper.run_algo(name, data_from_numpy(backend, data), *args, **kwargs)

    def fit(self, algo='nuts', **kwargs):
        """
        Fits the wrapped model.

        :param algo: The algorithm used for inference, either ``'nuts'``, ``'svi'`` or ``'prior'``.
        :type algo: str
        :param kwargs: Inference algorithm-specific keyword arguments. See :func:`~brmp.ModelAndData.nuts`,
                       :func:`~brmp.ModelAndData.svi` and :func:`~brmp.ModelAndData.prior` for details.
        :return: A model fit.
        :rtype: brmp.fit.Fit

        Example::

          fit = brm('y ~ x', df).fit()

        """
        assert algo in ['prior', 'nuts', 'svi']
        return getattr(self, algo)(**kwargs)

    def nuts(self, iter=10, warmup=None, num_chains=1, seed=None, backend=numpyro_backend):
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
        :param seed: Random seed.
        :type seed: int
        :param backend: The backend used to perform inference.
        :type backend: brmp.backend.Backend
        :return: A model fit.
        :rtype: brmp.fit.Fit

        Example::

          fit = brm('y ~ x', df).nuts()

        """
        warmup = iter // 2 if warmup is None else warmup
        return self.run_algo('nuts', backend, iter, warmup, num_chains, seed)

    def svi(self, iter=10, num_samples=10, seed=None, backend=pyro_backend, **kwargs):
        """
        Fit the model using stochastic variational inference.

        :param iter: The number of optimisation steps to take.
        :type iter: int
        :param num_samples: The number of samples to take from the variational
                            posterior.
        :type num_samples: int
        :param seed: Random seed.
        :type seed: int
        :param backend: The backend used to perform inference.
        :type backend: brmp.backend.Backend
        :return: A model fit.
        :rtype: brmp.fit.Fit

        Example::

          fit = brm('y ~ x', df).svi()

        """
        return self.run_algo('svi', backend, iter, num_samples, seed, **kwargs)

    def prior(self, num_samples=10, seed=None, backend=pyro_backend):
        """
        Sample from the prior.

        :param num_samples: The number of samples to take.
        :type num_samples: int
        :param seed: Random seed.
        :type seed: int
        :param backend: The backend used to perform inference.
        :type backend: brmp.backend.Backend

        :return: A model fit.
        :rtype: brmp.fit.Fit

        Example::

          fit = brm('y ~ x', df).prior()

        """
        return self.run_algo('prior', backend, num_samples, seed)

    def __repr__(self):
        return model_repr(self.model.desc)
