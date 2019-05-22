from collections import namedtuple

import torch
import numpy as np
import pandas as pd
# http://pandas.pydata.org/pandas-docs/stable/reference/general_utility_functions.html#dtype-introspection
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

from pyro.contrib.brm.utils import join
from .formula import Formula, Intercept


# TODO: Refer to dataframe metadata as 'schema' in order to avoid
# confusion with the similarly named design matrix metadata?
def make_metadata_lookup(metadata):
    assert type(metadata) == list
    assert all(type(factor) == Factor for factor in metadata)
    # Turn a list of factors into a dictionary keyed by column name.
    return dict((factor.name, factor) for factor in metadata)

Factor = namedtuple('Factor',
                    ['name',    # column name
                     'levels']) # list of level names

# Extract metadata (as expected by `genmodel`) from a pandas
# dataframe.
def dfmetadata(df):
    return [Factor(c, list(df[c].dtype.categories))
            for c in df
            if is_categorical_dtype(df[c])]

# TODO: Use the result of `coding` to generate more realistic dummy
# data. i.e. Rather than just having X & Z matrices of the correct
# size, intercepts can be all ones, and categorical columns can be
# appropriately coded (width `codefactor`) random data.
def dummydata(formula, metadata, N):
    import torch
    assert type(metadata) == dict
    data = {}
    M = width(formula.pterms, metadata)
    data['X'] = torch.rand(N, M)
    for i, group in enumerate(formula.groups):
        M_i = width(group.gterms, metadata)
        num_levels = len(metadata[group.column].levels)
        data['Z_{}'.format(i+1)] = torch.rand(N, M_i)
        # Maps (indices of) data points to (indices of) levels.
        data['J_{}'.format(i+1)] = torch.randint(0, num_levels, size=[N])
    data['y_obs'] = torch.rand(N)
    return data

# --------------------
# Design matrix coding
# --------------------

def codenumeric(dfcol):
    assert is_numeric_dtype(dfcol)
    return [dfcol]

# Codes a categorical column/factor. When reduced==False the column is
# dummy/one-of-K coded.

# x = [A, B, C, A]

# x0 x1 x2
#  1  0  0
#  0  1  0
#  0  0  1
#  1  0  0

# When reduced==True the same coding is used, but the first column is
# dropped.

# x1 x2
#  0  0
#  1  0
#  0  1
#  0  0

def codefactor(dfcol, reduced):
    assert is_categorical_dtype(dfcol)
    factors = dfcol.cat.categories
    num_levels = len(factors)
    start = 1 if reduced else 0
    return [dfcol == factors[i] for i in range(start, num_levels)]

def col2torch(col):
    if type(col) == torch.Tensor:
        assert col.dtype == torch.float32
        return col
    else:
        # TODO: It's possible to do torch.tensor(col) here. What does
        # that do? Is it preferable to this?
        return torch.from_numpy(col.to_numpy(np.float32))

def term_order(term):
    return 0 if type(term) == Intercept else 1


InterceptC = namedtuple('InterceptC', [])
CategoricalC = namedtuple('CategoricalC', ['factor', 'reduced'])
NumericC = namedtuple('NumericC', ['name'])

# TODO: I do similar dispatching on type in `designmatrix` and
# `designmatrix_metadata`. It would be more Pythonic to turn
# InterceptC etc. into classes with `width` and `code` methods.

def widthC(c):
    if type(c) in [InterceptC, NumericC]:
        return 1
    elif type(c) == CategoricalC:
        return len(c.factor.levels) - (1 if c.reduced else 0)
    else:
        raise Exception('Unknown coding type.')

# Generates a description of how the given terms ought to be coded
# into a design matrix.
def coding(terms, metadata):
    assert type(terms) == list
    assert type(metadata) == dict
    def code(i, term):
        assert type(term) in [str, Intercept]
        if type(term) == Intercept:
            return InterceptC()
        elif term in metadata:
            factor = metadata[term]
            return CategoricalC(factor, reduced=i>0)
        else:
            return NumericC(term)
    sorted_terms = sorted(terms, key=term_order)
    return [code(i, term) for i, term in enumerate(sorted_terms)]

def width(terms, metadata):
    assert type(metadata) == dict
    return sum(widthC(c) for c in coding(terms, metadata))

# Build a simple design matrix (as a torch tensor) from columns of a
# pandas data frame.

# TODO: There ought to be a check somewhere to ensure that all terms
# are either numeric or categorical. We're making this assumption in
# `coding`, where anything not mentioned in dfmetadata(df) is assumed
# to be numeric. But we can't check it there because we don't have a
# concreate dataframe at that point. Here we have a list of terms and
# the df, so this seems like a good place. An alternative is to do
# this in makedata, but we'd need to do so for population and group
# levels.

def designmatrix(terms, df):
    assert type(terms) == list
    N = len(df)
    def dispatch(code):
        if type(code) == InterceptC:
            return [torch.ones(N, dtype=torch.float32)]
        elif type(code) == CategoricalC:
            return codefactor(df[code.factor.name], code.reduced)
        elif type(code) == NumericC:
            return codenumeric(df[code.name])
        else:
            raise Exception('Unknown coding type.')
    metadata = make_metadata_lookup(dfmetadata(df))
    coding_desc = coding(terms, metadata)
    coded_cols = join([dispatch(c) for c in coding_desc])
    X = torch.stack([col2torch(col) for col in coded_cols], dim=1) if coded_cols else torch.empty(N, 0)
    assert X.shape == (N, width(terms, metadata))
    #print(designmatrix_metadata(terms, df))
    return X

# --------------------------------------------------
# Experimenting with design matrix metadata:
#
# `designmatrix_metadata` computes a list of readable design matrix
# column names. The idea is that (following brms) this information has
# a number of uses:
#
# - Improve readability of e.g. `Fit` summary. e.g. Instead of just
#   showing the `b` vector, we can use this information to identify
#   each coefficient.
#
# - Users need to be able to specify their own priors for individual
#   coefficients. I think this information is used as the basis of
#   that, allowing priors to be specified by coefficient name?

def numeric_metadata(code):
    return [code.name]

def categorical_metadata(code):
    start = 1 if code.reduced else 0
    return ['{}[{}]'.format(code.factor.name, cat)
            for cat in code.factor.levels[start:]]

def designmatrix_metadata(terms, metadata):
    assert type(terms) == list
    def dispatch(code):
        if type(code) == InterceptC:
            return ['intercept']
        elif type(code) == CategoricalC:
            return categorical_metadata(code)
        elif type(code) == NumericC:
            return numeric_metadata(code)
        else:
            raise Exception('Unknown coding type.')
    coding_desc = coding(terms, metadata)
    return join([dispatch(c) for c in coding_desc])

DesignMeta = namedtuple('DesignMeta', 'population groups')
PopulationMeta = namedtuple('PopulationMeta', 'coefs')
GroupMeta = namedtuple('GroupMeta', 'name coefs')

def designmatrices_metadata(formula, metadata):
    p = PopulationMeta(designmatrix_metadata(formula.pterms, metadata))
    gs = [GroupMeta(group.column, designmatrix_metadata(group.gterms, metadata))
          for group in formula.groups]
    return DesignMeta(p, gs)

# --------------------------------------------------

def lookupvector(column, df):
    assert type(column) == str
    assert type(df) == pd.DataFrame
    assert column in df
    assert is_categorical_dtype(df[column])
    return torch.from_numpy(df[column].cat.codes.to_numpy(np.int64))

def responsevector(column, df):
    assert type(column) == str
    assert type(df) == pd.DataFrame
    assert column in df
    dfcol = df[column]
    if is_numeric_dtype(dfcol):
        coded = codenumeric(dfcol)
    elif is_categorical_dtype(dfcol) and len(dfcol.cat.categories) == 2:
        # TODO: How does a user know how this was coded? For design
        # matrices this is revealed by the column names in the design
        # metadata, but we don't have the here.
        coded = codefactor(dfcol, reduced=True)
    else:
        raise Exception('Don\'t know how to code a response of this type.')
    assert len(coded) == 1
    return col2torch(coded[0])

def makedata(formula, df):
    assert type(formula) == Formula
    assert type(df) == pd.DataFrame
    data = {}
    data['X'] = designmatrix(formula.pterms, df)
    data['y_obs'] = responsevector(formula.response, df)
    for i, group in enumerate(formula.groups):
        data['Z_{}'.format(i+1)] = designmatrix(group.gterms, df)
        data['J_{}'.format(i+1)] = lookupvector(group.column, df)
    return data
