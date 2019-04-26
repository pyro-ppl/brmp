from collections import namedtuple

import torch
import numpy as np
import pandas as pd
# http://pandas.pydata.org/pandas-docs/stable/reference/general_utility_functions.html#dtype-introspection
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

from .formula import Formula, Intercept

def make_metadata_lookup(metadata):
    assert type(metadata) == list
    assert all(type(factor) == Factor for factor in metadata)
    # Turn a list of factors into a dictionary keyed by column name.
    return dict((factor.name, factor) for factor in metadata)

# Computes the number of entries a column adds to the design matrix.
# Each numeric column contributes 1 entry. Each factor contributes
# num_levels-1.

# The latter is because a factor with 4 levels (for example) is coded
# like so in the design matrix:

# x = factor(c(0, 1, 2, 3))
#
# Intercept x1 x2 x3
# 1          0  0  0
# 1          1  0  0
# 1          0  1  0
# 1          0  0  1

# This is always the case when an intercept is present, otherwise
# things are a little more subtle. Without an intercept, the factor
# above would be coded like so if it appears as the first term in e.g.
# pterms.

# x0 x1 x2 x3
#  1  0  0  0
#  0  1  0  0
#  0  0  1  0
#  0  0  0  1

# Subsequent factors are then coded as they would be were an intercept
# present.

Factor = namedtuple('Factor',
                    ['name',        # column name
                     'num_levels']) # number of levels

# Extract metadata (as expected by `genmodel`) from a pandas
# dataframe.
def dfmetadata(df):
    return [Factor(c, len(df[c].dtype.categories))
            for c in df
            if type(df[c].dtype) == pd.CategoricalDtype]

# TODO: Factor out the logic for figuring out how big each matrix is.
# Could re-use here and when generating model code? (When generating
# dimension checking code, constants N,M, parameters for model
# function.)
def dummydata(formula, metadata, N):
    import torch
    metadata_lookup = make_metadata_lookup(metadata)
    data = {}
    M = width(formula.pterms, metadata_lookup)
    data['X'] = torch.rand(N, M)
    for i, group in enumerate(formula.groups):
        M_i = width(group.gterms, metadata_lookup)
        num_levels = metadata_lookup[group.column].num_levels
        data['Z_{}'.format(i+1)] = torch.rand(N, M_i)
        # Maps (indices of) data points to (indices of) levels.
        data['J_{}'.format(i+1)] = torch.randint(0, num_levels, size=[N])
    data['y_obs'] = torch.rand(N)
    return data

# --------------------
# Design matrix coding
# --------------------

def codenumeric(dfcol):
    return [dfcol]

def codefactor(dfcol, reduced):
    factors = dfcol.cat.categories
    num_levels = len(factors)
    start = 1 if reduced else 0
    return [dfcol == factors[i] for i in range(start, num_levels)]

def join(lists):
    return sum(lists, [])

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

def widthC(c):
    if type(c) in [InterceptC, NumericC]:
        return 1
    elif type(c) == CategoricalC:
        return c.factor.num_levels - (1 if c.reduced else 0)
    else:
        raise Exception('Unknown coding type.')

# Generates a description of how the given terms ought to be coded
# into a design matrix.
def coding(terms, metadata_lookup):
    assert type(terms) == list
    def code(i, term):
        assert type(term) in [str, Intercept]
        if type(term) == Intercept:
            return InterceptC()
        elif term in metadata_lookup:
            factor = metadata_lookup[term]
            return CategoricalC(factor, reduced=i>0)
        else:
            return NumericC(term)
    sorted_terms = sorted(terms, key=term_order)
    return [code(i, term) for i, term in enumerate(sorted_terms)]

def width(terms, metadata_lookup):
    return sum(widthC(c) for c in coding(terms, metadata_lookup))

# TODO: brms keeps track of the meaning of each column, and uses that
# when e.g. presenting summaries. Collect that information here?

# Build a simple design matrix (as a torch tensor) from columns of a
# pandas data frame.
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
    metadata_lookup = make_metadata_lookup(dfmetadata(df))
    coding_desc = coding(terms, metadata_lookup)
    coded_cols = join([dispatch(c) for c in coding_desc])
    X = torch.stack([col2torch(col) for col in coded_cols], dim=1) if coded_cols else torch.empty(N, 0)
    assert X.shape == (N, width(terms, metadata_lookup))
    return X


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
    assert is_numeric_dtype(df[column])
    return torch.from_numpy(df[column].to_numpy(np.float32))

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
