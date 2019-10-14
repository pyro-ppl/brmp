import itertools
import operator as op
import random
from collections import OrderedDict, namedtuple
from functools import reduce

import numpy as np
import pandas as pd
# http://pandas.pydata.org/pandas-docs/stable/reference/general_utility_functions.html#dtype-introspection
from pandas.api.types import (is_categorical_dtype, is_float_dtype,
                              is_integer_dtype)

from brmp.formula import Formula, OrderedSet, Term, allfactors
from brmp.utils import join

# TODO: Levels of pandas categorical columns can be any hashable type
# I think. Is our implementation flexible enough to handle the same?
# If not, we ought to check the type here and throw an error when it's
# something we can't handle. Two areas to consider are whether it's
# possible to specify priors on coefs arising from levels of a factor
# (I think this works if the user knows how the types values are
# turned into strings), and whether posterior summaries look OK. (Both
# of which have to do with whether instances of the type can be
# converted to strings in a sensible way.)

Categorical = namedtuple('Categorical',
                         ['name',     # column name
                          'levels'])  # list of level names

Integral = namedtuple('Integral',
                      ['name',
                       'min',
                       'max'])


class RealValued(namedtuple('RealValued', ['name', 'min', 'max'])):
    def __new__(cls, name, min=-np.inf, max=np.inf):
        assert type(min) == float
        assert type(max) == float
        assert max >= min
        return super(RealValued, cls).__new__(cls, name, min, max)


def is_numeric_col(col):
    assert type(col) in [Categorical, Integral, RealValued]
    return not type(col) == Categorical


# TODO: This is rather a long way from idiomatic Python code.
# Re-implement Metadata as a class, perhaps with subclasses for
# instances from df vs. manually specified columns.

# I only know for sure that I'll need the *number* of levels, and not
# a list of the all of the levels themselves. However, I suspect this
# *will* be useful when it comes to checking that and "new data" given
# to `fitted` is compatible with the model/previous data. If this
# turns out not to be the case, I might want to change this.

# columns: A list of RealValued, Categorical, Integral values.
#
# column: Find a single column from columns by name.
#
# levels: Given a list of categorical column names, returns all of the
# combinations of the individual column levels that actually appear in
# the data. These are ordered according to the Cartesian product of
# the individual column levels.
#
Metadata = namedtuple('Metadata', 'columns column levels')


def make_column_lookup(columns):
    assert type(columns) == list
    assert all(type(col) in [Categorical, Integral, RealValued] for col in columns)
    # Turn a list of columns into a dictionary keyed by column name.
    return dict((col.name, col) for col in columns)


def df_levels(columns, df):
    assert type(columns) == list
    assert all(type(col) == str for col in columns)
    assert type(df) == pd.DataFrame
    assert all(col in df and is_categorical_dtype(df[col]) for col in columns)
    vals = [tuple(row) for _, row in df[columns].iterrows()]
    present = set(vals)
    all_possible_vals = list(itertools.product(*[df[col].cat.categories for col in columns]))
    table = [val for val in all_possible_vals if val in present]
    return table


# Extract column information from a pandas dataframe.
def dfcols(df):
    def dispatch(dfcol):
        if is_categorical_dtype(dfcol):
            return Categorical(dfcol.name, list(dfcol.dtype.categories))
        elif is_integer_dtype(dfcol):
            return Integral(dfcol.name, min(dfcol), max(dfcol))
        elif is_float_dtype(dfcol):
            return RealValued(dfcol.name, min(dfcol), max(dfcol))
        else:
            raise Exception('unhandled column type encountered for column "{}"'.format(dfcol.name))
    return [dispatch(df[c]) for c in df]


def metadata_from_df(df):
    assert type(df) == pd.DataFrame
    cols = dfcols(df)
    lu = make_column_lookup(cols)
    return Metadata(cols, lambda name: lu[name], lambda names: df_levels(names, df))


def all_levels(names, metadata_lookup):
    assert type(names) == list
    assert all(type(name) == str for name in names)
    assert type(metadata_lookup) == dict
    assert all(name in metadata_lookup and type(metadata_lookup[name]) == Categorical for name in names)
    all_possible_vals = list(itertools.product(*[metadata_lookup[name].levels for name in names]))
    return all_possible_vals


# Makes the assumption that all possible levels are present. The idea
# is that this is sufficient to allow us to generate model code from
# only a description of the columns, even when there's a `g1:g2` like
# grouping term in the model. However, there's no gaurentee that a
# random data frame generated from this will contain all of these
# levels.
def metadata_from_cols(cols):
    assert type(cols) == list
    assert all(type(col) in [RealValued, Integral, Categorical] for col in cols)
    lu = make_column_lookup(cols)
    return Metadata(cols, lambda name: lu[name], lambda names: all_levels(names, lu))


def dummy_df(cols, N):
    assert type(cols) == list
    assert all(type(col) in [RealValued, Categorical, Integral] for col in cols)

    def gen_numeric_col(factor):
        if np.isfinite(factor.min) and np.isfinite(factor.max):
            return np.random.uniform(factor.min, factor.max, N)
        elif np.isfinite(factor.min):
            return factor.min + np.abs(np.random.randn(N))
        elif np.isfinite(factor.max):
            return factor.max - np.abs(np.random.randn(N))
        else:
            return np.random.randn(N)

    def gen_categorical_col(levels):
        return pd.Categorical(random.choices(levels, k=N))

    def gen_integral_col(factor):
        return np.random.randint(factor.min, factor.max + 1, N)

    def dispatch(col):
        if type(col) == RealValued:
            return gen_numeric_col(col)
        elif type(col) == Categorical:
            return gen_categorical_col(col.levels)
        elif type(col) == Integral:
            return gen_integral_col(col)
        else:
            raise Exception('unknown factor type')

    return pd.DataFrame({col.name: dispatch(col) for col in cols})


def dummy_design(formula, metadata, N):
    assert type(formula) == Formula
    assert type(metadata) == dict
    assert set(allfactors(formula)).issubset(metadata.keys())
    return makedata(formula, dummy_df(metadata, N))

# --------------------
# Design matrix coding
# --------------------


# A version of product in which earlier elements of the returned
# tuples vary more rapidly than later ones. This matches the way
# interactions are coded in R.
def product(iterables):
    return [tuple(reversed(t)) for t in itertools.product(*reversed(iterables))]


# Taken from the itertools documentation.
def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


# A Term represents the interaction between zero or more factors. This
# function describes how the coding for such a term can be performed
# by coding multiple interactions, each using a reduced rank coding.
# For example:
#
# Term(<'a'>) can be coded as [Intercept, a (reduced)].
# Term(<'a','b'>) can be coded as [Intercept, a (reduced), b (reduced), a:b (reduced)].
def decompose(term):
    assert type(term) == Term
    return [tuple(CategoricalCoding(factor, True) for factor in subset) for subset in powerset(term.factors)]


# Attempt to absorb t2 into t1. If this is possible the result of
# doing so is returned. Otherwise None is returned. This rule explains
# when absorbtion is possible.

#  t2    t1                  result
# { X , X U {x-} , ...} == { X U {x+} , ...}

# I'm maintaining the original order here, even though Patsy doesn't.
# The reason I am doing so (by keeping tuples rather than sets) is
# that when we come it add columns to the design matrix, it makes a
# difference whether we code {a+,b+} as:
#
# a[a1]:b[b1]  a[a2]:b[b1]  a[a1]:b[b2]  a[a2]:b[b2]
#
# or:
#
# b[b1]:a[a1]  b[b2]:a[a1]  b[b1]:a[a2]  b[b2]:a[a2]
#
# These are the same columns but they appear in a different order. If
# we treat the interactions as sets (not maintaining the order given
# by the formula) then it seems that the order in which the columns
# appear is going to be determined by whatever the set implementation
# gives us when eventually turn it into a list. This seems
# problematic, since the design matrix would not be invariant under
# renaming of the factors (i.e. variables) in the formula. (Or worse,
# it might even be non-deterministic.)
#
# Note that I haven't observed this undesirable behaviour in Patsy.
# However, I have noticed that some order information *is* tracked, as
# can be seen when printing a design matrix. Doing so might show
# something like "'a:b' (columns 0:4), 'a:b:c' (columns 4:8)" for
# example. So, even though `_Subterm` (the analogue of one of my
# tuples of CategoricalCodings) is set based, it's possible that this is
# used to recover the original order.

# TODO: Would it be clearer or more efficient to use OrderedSet rather
# than tuple here? I'm not sure.

def absorb(t1, t2):
    assert type(t1) == tuple
    assert all(type(p) == CategoricalCoding for p in t1)
    assert type(t2) == tuple
    assert all(type(p) == CategoricalCoding for p in t2)
    s1 = set(t1)
    s2 = set(t2)
    if s2.issubset(s1) and len(s1) - len(s2) == 1:
        diff = s1.difference(s2)
        assert len(diff) == 1
        extra_factor = list(diff)[0]
        if extra_factor.reduced:
            factor = CategoricalCoding(extra_factor.factor, False)
            return tuple((factor if f == extra_factor else f) for f in t1)


def simplify_one(termcoding):
    assert type(termcoding) == list
    assert all(type(t) == tuple and all(type(p) == CategoricalCoding for p in t) for t in termcoding)
    for i, j in itertools.permutations(range(len(termcoding)), 2):
        newterm = absorb(termcoding[i], termcoding[j])
        if newterm:
            out = termcoding[:]
            out[i] = newterm  # Replace with absorbing interaction.
            del out[j]        # Remove absorbed interaction.
            return out


def simplify(termcoding):
    assert type(termcoding) == list
    assert all(type(t) == tuple and all(type(p) == CategoricalCoding for p in t) for t in termcoding)
    while True:
        maybe_termcoding = simplify_one(termcoding)
        if maybe_termcoding is None:
            return termcoding  # We're done.
        termcoding = maybe_termcoding


# all_previous([['a'], ['b','c'], ['d']])
# ==           [{},    {'a'},     {'a','b','c'}]
def all_previous(xss):
    if len(xss) == 0:
        return []
    else:
        return [set()] + [set(xss[0]).union(xs) for xs in all_previous(xss[1:])]


# This is an attempt to implement the algorithm described here:
# https://patsy.readthedocs.io/en/latest/formulas.html#technical-details

def code_categorical_terms(terms):
    # It is assumed that each element of `terms` describes an
    # interaction between zero or more categorical factors.
    decomposed = [decompose(t) for t in terms]
    non_redundant = [[t for t in term if t not in previous]
                     for term, previous in zip(decomposed, all_previous(decomposed))]
    return [simplify(t) for t in non_redundant]


def partition(pred, iterable):
    t1, t2 = itertools.tee(iterable)
    return list(itertools.filterfalse(pred, t1)), list(filter(pred, t2))


CategoricalCoding = namedtuple('CategoricalCoding', 'factor reduced')
CategoricalCoding.__repr__ = lambda self: '{}{}'.format(self.factor, '-' if self.reduced else '+')

NumericCoding = namedtuple('NumericCoding', ['factor'])
NumericCoding.__repr__ = lambda self: self.factor


# Codes a group of terms that all share a common set of numeric factors.
# Returns a list of coded interactions.

def code_group_of_terms(terms, shared_numeric_factors):
    assert type(terms) == list
    assert all(type(term) == Term for term in terms)
    assert type(shared_numeric_factors) == OrderedSet

    # It's also the case that each term should contain no numeric
    # factors not mentions in `shared_numeric_factors`, but that is
    # not checked here.
    assert all(all((factor in term.factors) for factor in shared_numeric_factors) for term in terms)

    def drop_numeric_factors(term):
        factors = [f for f in term.factors if f not in shared_numeric_factors]
        return Term(OrderedSet(*factors))

    categorical_terms = [drop_numeric_factors(term) for term in terms]
    codings_for_terms = code_categorical_terms(categorical_terms)

    num_codings_dict = {f: NumericCoding(f) for f in shared_numeric_factors}

    # This adds codings for the shared numeric factors to the coding
    # of a categorical interaction, respecting the factor order in the
    # source term.
    #
    # e.g. term   = Term(<a,x,b>)
    #      coding = (b+,)
    # Returns:
    #      (x,b+)
    # (Assuming shared numeric factors is ['x'].)
    #
    def extend_with_numeric_factors(term, coding):
        cat_codings_dict = {c.factor: c for c in coding}
        # This gives us a dictionary that maps from factor names
        # (factors in coding U shared numeric factors) to codings
        # (e.g. CategoricalCoding, NumericCoding).
        codings_dict = dict(cat_codings_dict, **num_codings_dict)
        # We then grab all of these codings following the factor order
        # in the term. (Note that some factors in the term may not
        # appear in the coding.)
        out = [codings_dict[f] for f in term.factors if f in codings_dict]
        assert len(out) == len(codings_dict)
        return out

    assert len(terms) == len(codings_for_terms)  # complain if zip will drop things
    return join([[extend_with_numeric_factors(term, coding) for coding in codings]
                 for (term, codings) in zip(terms, codings_for_terms)])


# [('a', 100), ('b', 200), ('a', 300)] =>
# {'a': [100, 300], 'b': [200]}
def group(pairs):
    assert type(pairs) == list
    assert all(type(pair) == tuple and len(pair) == 2 for pair in pairs)
    # Remember insertion order. i.e. The returned dictionary captures
    # the order in which the groups were first encountered in the
    # input list.
    out = OrderedDict()
    for (k, v) in pairs:
        if k not in out:
            out[k] = []
        out[k].append(v)
    return out


# Partition terms by the numeric factors they contain, and sort the
# resulting groups.
def partition_terms(terms, metadata):
    assert type(terms) == OrderedSet
    assert type(metadata) == Metadata

    def numeric_factors(term):
        factors = [f for f in term.factors if is_numeric_col(metadata.column(f))]
        return OrderedSet(*factors)

    # The idea here is to store the full term (including the numeric
    # factors) as a way of remembering the order in which the numeric
    # and numeric factors originally appeared. I think Patsy does
    # something like this.
    groups = group([(numeric_factors(term), term) for term in terms])
    # Sort the groups. First comes the group containing no numeric
    # factors. The remaining groups appear in the order in which a
    # term containing exactly those numeric factors associated with
    # the group first appears in `terms`. (The latter is guaranteed by
    # the fact that `group` is order aware.
    empty_set = OrderedSet()
    first, rest = partition(lambda kv: kv[0] != empty_set, groups.items())
    return first + rest


def code_lengths(contrasts):
    assert type(contrasts) == dict
    return {k: mat.shape[1] for k, mat in contrasts.items()}


# Build a simple design matrix (as a torch tensor) from columns of a
# pandas data frame.

# TODO: This could be generalised to work with a wider range of data
# representation. Rather than expecting a pandas dataframe, it could
# take a dictionary that gives access to the columns (iterables full
# of floats, ints, or level values?) and the existing dataframe
# metadata structure to describe the types of the columns, etc.

def designmatrix(terms, df, metadata, contrasts):
    assert type(terms) == OrderedSet
    coded_interactions = code_terms(terms, metadata)
    product_cols = join(coded_interaction_to_product_cols(code, metadata, code_lengths(contrasts))
                        for code in coded_interactions)
    N = len(df)
    arrs = [execute_product_col(pcol, df, metadata, contrasts) for pcol in product_cols]
    X = np.stack(arrs, axis=1) if arrs else np.empty((N, 0))
    assert X.shape[0] == N
    if X.shape[0] > 0 and X.shape[1] > 0 and np.linalg.matrix_rank(X) != X.shape[1]:
        print('WARNING: Design matrix may not be full rank.')
    return X


# Take an ordered set of terms (e.g. from a formula) to a list of
# coded interctions.
# e.g. code_terms(parse('y ~ 1 + a:b').terms, metadata) => .?.

# Terms with in a group are ordered by their order, i.e. the number of
# factors they contain.
def sort_by_order(terms):
    return sorted(terms, key=lambda term: len(term.factors))


def code_terms(terms, metadata):
    assert type(metadata) == Metadata
    groups = partition_terms(terms, metadata)
    return join(code_group_of_terms(sort_by_order(terms), shared_num_factors)
                for shared_num_factors, terms in groups)


# TODO: Arg. checks for these named tuples?

IndicatorCol = namedtuple('IndicatorCol', ['factor', 'level'])
IndicatorCol.__repr__ = lambda self: 'I[{}={}]'.format(self.factor, self.level)

CustomCol = namedtuple('CustomCol', ['factor', 'index'])
CustomCol.__repr__ = lambda self: 'Custom({})[{}]'.format(self.factor, self.index)

NumericCol = namedtuple('NumericCol', ['factor'])
NumericCol.__repr__ = lambda self: 'Num({})'.format(self.factor)

# Represents the product of zero of more columns.
ProductCol = namedtuple('ProductCol', ['cols'])  # `cols` is expected to be a list


def coded_interaction_to_product_cols(coded_interaction, metadata, code_lengths):
    assert type(coded_interaction) == list
    assert type(metadata) == Metadata
    assert type(code_lengths) == dict
    assert all(type(c) in [CategoricalCoding, NumericCoding] for c in coded_interaction)

    cs, ns = partition(lambda cf: type(cf) == NumericCoding, coded_interaction)

    def go(c):
        # Patsy and R seem to differ in what they do here. This
        # implementation is similar to Patsy. In contrast, in R the
        # custom coding is only used when `c.reduced == True`.
        if c.factor in code_lengths:
            # Custom coding.
            code_length = code_lengths[c.factor]
            assert type(code_length) == int
            return [CustomCol(c.factor, i) for i in range(code_length)]
        else:
            # Default coding.
            all_levels = metadata.column(c.factor).levels
            levels = all_levels[1:] if c.reduced else all_levels
            return [IndicatorCol(c.factor, level) for level in levels]

    interactions = product([go(c) for c in cs])

    ncols_dict = {n.factor: NumericCol(n.factor) for n in ns}

    def extend_with_numeric_cols(ccols):
        ccols_dict = {ccol.factor: ccol for ccol in ccols}
        cols_dict = dict(ccols_dict, **ncols_dict)
        # Make a list of both the indicator and numeric columns,
        # ordered by the factor order in the coded interaction given
        # as input.
        out = [cols_dict[ci.factor] for ci in coded_interaction]
        assert len(out) == len(coded_interaction)
        return out

    return [ProductCol(extend_with_numeric_cols(ccols)) for ccols in interactions]


def product_col_to_coef_name(product_col):
    assert type(product_col) == ProductCol

    # TODO: I do similar dispatching elsewhere. It would be more
    # Pythonic to turn `IndicatorCol` etc. into classes appropriate
    # methods. e.g. `coef_name` in this case.

    def dispatch(col):
        if type(col) == IndicatorCol:
            return '{}[{}]'.format(col.factor, col.level)
        elif type(col) == NumericCol:
            return col.factor
        elif type(col) == CustomCol:
            return '{}[custom.{}]'.format(col.factor, col.index)
        else:
            raise Exception('unknown column type')

    if len(product_col.cols) == 0:
        return 'intercept'
    else:
        return ':'.join(dispatch(col) for col in product_col.cols)


def execute_product_col(product_col, df, metadata, contrasts):
    assert type(product_col) == ProductCol
    assert type(df) == pd.DataFrame

    def dispatch(col):
        dfcol = df[col.factor]
        if type(col) == IndicatorCol:
            assert is_categorical_dtype(dfcol)
            return (dfcol == col.level).to_numpy()
        elif type(col) == NumericCol:
            assert is_float_dtype(dfcol) or is_integer_dtype(dfcol)
            return dfcol.to_numpy()
        elif type(col) == CustomCol:
            assert col.factor in contrasts
            mat = contrasts[col.factor]
            levels = metadata.column(col.factor).levels
            # TODO: This can be triggered in normal use, so turn into
            # friendly error. It probably makes sense to check for
            # this earlier. This could possibly happen in `defm` after
            # creating the metadata, though this would also require a
            # separate check for SequentialOED. Is there anywhere
            # sensible to put this that catches both?
            assert len(levels) == mat.shape[0]
            assert col.index < mat.shape[1]
            # TODO: Better asymptotics then using `.index()`
            out = mat[[levels.index(val) for val in dfcol], col.index]
            return out
        else:
            raise Exception('unknown column type')

    # We use a vector of ones as the initial value of the reduction.
    # This is inefficient, but OK for now. This gives the correct
    # behaviour of the intercept column and will also ensure that
    # indicator columns are returned at floats even when there are no
    # numeric columsn in the product.

    N = len(df)
    init = np.ones(N)
    arr = reduce(op.mul, (dispatch(col) for col in product_col.cols), init)
    assert arr.dtype == np.float64
    assert len(arr.shape) == 1
    assert arr.shape[0] == N
    return arr


def coef_names(terms, metadata, code_lengths):
    assert type(terms) == OrderedSet
    assert type(metadata) == Metadata
    coded_interactions = code_terms(terms, metadata)
    product_cols = join(coded_interaction_to_product_cols(code, metadata, code_lengths)
                        for code in coded_interactions)
    return [product_col_to_coef_name(pcol) for pcol in product_cols]


# --------------------------------------------------

def lookupvector(columns, df, metadata):
    assert type(columns) == list
    assert all(type(col) == str for col in columns)
    assert type(df) == pd.DataFrame
    # TODO: Perhaps better to use a dictionary in case the number of
    # combinations present in the data becomes large.
    table = metadata.levels(columns)
    # For each row in the data, look up its combination of grouping
    # columns values in the table, using the position of the matching
    # row as the index.
    indices = [table.index(tuple(row)) for _, row in df[columns].iterrows()]
    return np.array(indices, dtype=np.int64)


def responsevector(column, df, metadata):
    assert type(column) == str
    assert type(df) == pd.DataFrame
    assert column in df
    col = metadata.column(column)
    if type(col) == RealValued or type(col) == Integral:
        code = NumericCoding(column)
    elif type(col) == Categorical and len(col.levels) == 2:
        # TODO: How does a user know how this was coded? For design
        # matrices this is revealed by the column names in the design
        # metadata, but we don't have the here.
        code = CategoricalCoding(column, True)
    else:
        raise Exception('Don\'t know how to code a response of this type.')
    pcols = coded_interaction_to_product_cols([code], metadata, {})
    assert len(pcols) == 1
    return execute_product_col(pcols[0], df, metadata, {})


def predictors(formula, df, metadata, contrasts):
    assert type(formula) == Formula
    assert type(df) == pd.DataFrame
    data = {}
    data['X'] = designmatrix(formula.terms, df, metadata, contrasts)
    for i, group in enumerate(formula.groups):
        data['Z_{}'.format(i)] = designmatrix(group.terms, df, metadata, contrasts)
        data['J_{}'.format(i)] = lookupvector(group.columns, df, metadata)
    return data


def makedata(formula, df, metadata, contrasts):
    return dict(predictors(formula, df, metadata, contrasts),
                y_obs=responsevector(formula.response, df, metadata))
