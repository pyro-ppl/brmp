from collections import namedtuple, OrderedDict
import itertools
from functools import reduce
import operator as op
import random

import numpy as np
import pandas as pd
# http://pandas.pydata.org/pandas-docs/stable/reference/general_utility_functions.html#dtype-introspection
from pandas.api.types import is_categorical_dtype, is_integer_dtype, is_float_dtype

from pyro.contrib.brm.utils import join
from pyro.contrib.brm.formula import Formula, OrderedSet, Term, allfactors


# TODO: Refer to dataframe metadata as 'schema' in order to avoid
# confusion with the similarly named design matrix metadata?
def make_metadata_lookup(metadata):
    assert type(metadata) == list
    assert all(type(factor) in [Categorical, Integral, RealValued] for factor in metadata)
    # Turn a list of factors into a dictionary keyed by column name.
    return dict((factor.name, factor) for factor in metadata)


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
                         ['name',    # column name
                          'levels']) # list of level names

Integral = namedtuple('Integral',
                      ['name',
                       'min',
                       'max'])

RealValued = namedtuple('RealValued', ['name'])

def is_numeric_col(col):
    assert type(col) in [Categorical, Integral, RealValued]
    return not type(col) == Categorical

# Extract metadata from a pandas dataframe.
def dfmetadata(df):
    def dfcol2meta(dfcol):
        if is_categorical_dtype(dfcol):
            return Categorical(dfcol.name, list(dfcol.dtype.categories))
        elif is_integer_dtype(dfcol):
            return Integral(dfcol.name, min(dfcol), max(dfcol))
        elif is_float_dtype(dfcol):
            return RealValued(dfcol.name)
        else:
            raise Exception('unhandled column type encountered for column "{}"'.format(dfcol.name))
    return [dfcol2meta(df[c]) for c in df]

def dummy_df(metadata, N):
    assert type(metadata) == dict
    def gen_numeric_col():
        return np.random.rand(N)
    def gen_categorical_col(levels):
        return pd.Categorical(random.choices(levels, k=N))
    def gen_integral_col(factor):
        return np.random.randint(factor.min, factor.max + 1, N)
    def dispatch(factor):
        if type(metadata[factor]) == RealValued:
            return gen_numeric_col()
        elif type(metadata[factor]) == Categorical:
            return gen_categorical_col(metadata[factor].levels)
        elif type(metadata[factor]) == Integral:
            return gen_integral_col(metadata[factor])
        else:
            raise Exception('unknown factor type')
    cols = {factor: dispatch(factor) for factor in metadata}
    return pd.DataFrame(cols)

def dummy_design(formula, metadata, N):
    assert type(formula) == Formula
    assert type(metadata) == dict
    assert set(allfactors(formula)).issubset(metadata.keys())
    return makedata(formula, dummy_df(metadata, N))

# --------------------
# Design matrix coding
# --------------------

def codenumeric(dfcol):
    assert is_float_dtype(dfcol) or is_integer_dtype(dfcol)
    return [dfcol]


# A version of product in which earlier elements of the returned
# tuples vary more rapidly than later ones. This matches the way
# interactions are coded in R.
def product(iterables):
    return [tuple(reversed(t)) for t in itertools.product(*reversed(iterables))]

def codeindicator(dfcols, values):
    assert len(dfcols) == len(values)
    return reduce(op.and_,
                  (dfcol == value
                   for (dfcol, value) in zip(dfcols, values)))


def codeinteraction(dfcols, reduced_flags):
    assert type(dfcols) == list
    assert type(reduced_flags) == list
    assert len(dfcols) == len(reduced_flags)
    assert all(is_categorical_dtype(dfcol) for dfcol in dfcols)
    assert all(type(reduced) == bool for reduced in reduced_flags)

    # e.g. [('a1', 'b1'), ('a2', 'b1'), ...]
    #
    # where the first element of a tuple is a level from dfcol[0],
    # etc.
    cols = product([dfcol.cat.categories[1:] if reduced else dfcol.cat.categories
                    for dfcol, reduced in zip(dfcols, reduced_flags)])

    return [codeindicator(dfcols, values) for values in cols]

def codefactor(dfcol, reduced):
    return codeinteraction([dfcol], [reduced])

def col2numpy(col):
    default_dtype = np.float64
    if type(col) == np.ndarray:
        assert col.dtype == default_dtype
        return col
    else:
        return col.to_numpy(default_dtype)

InterceptC = namedtuple('InterceptC', [])
InteractionC = namedtuple('InteractionC', ['codes']) # codes is a list of CategoricalCs
CategoricalC = namedtuple('CategoricalC', ['factor', 'reduced'])
# Represents coding of either integer or real-valued columns.
NumericC = namedtuple('NumericC', ['name'])

# TODO: I do similar dispatching on type in `designmatrix` and
# `designmatrix_metadata`. It would be more Pythonic to turn
# InterceptC etc. into classes with `width` and `code` methods.

def widthC(c):
    if type(c) in [InterceptC, NumericC]:
        return 1
    elif type(c) == InteractionC:
        return reduce(
            op.mul,
            (len(code.factor.levels) - (1 if code.reduced else 0)
             for code in c.codes))
    else:
        raise Exception('Unknown coding type.')

CodedFactor = namedtuple('CodedFactor', 'factor reduced')
CodedFactor.__repr__ = lambda self: '{}{}'.format(self.factor, '-' if self.reduced else '')

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
    return [tuple(CodedFactor(factor, True) for factor in subset) for subset in powerset(term.factors)]


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
# tuples of CodedFactors) is set based, it's possible that this is
# used to recover the original order.

# TODO: Would it be clearer or more efficient to use OrderedSet rather
# than tuple here? I'm not sure.

def absorb(t1, t2):
    assert type(t1) == tuple
    assert all(type(p) == CodedFactor for p in t1)
    assert type(t2) == tuple
    assert all(type(p) == CodedFactor for p in t2)
    s1 = set(t1)
    s2 = set(t2)
    if s2.issubset(s1) and len(s1) - len(s2) == 1:
        diff = s1.difference(s2)
        assert len(diff) == 1
        extra_factor = list(diff)[0]
        if extra_factor.reduced:
            factor = CodedFactor(extra_factor.factor, False)
            return tuple((factor if f == extra_factor else f) for f in t1)

def simplify_one(termcoding):
    assert type(termcoding) == list
    assert all(type(t) == tuple and all(type(p) == CodedFactor for p in t) for t in termcoding)
    for i, j in itertools.permutations(range(len(termcoding)), 2):
        newterm = absorb(termcoding[i], termcoding[j])
        if newterm:
            out = termcoding[:]
            out[i] = newterm # Replace with absorbing interaction.
            del out[j]       # Remove absorbed interaction.
            return out

def simplify(termcoding):
    assert type(termcoding) == list
    assert all(type(t) == tuple and all(type(p) == CodedFactor for p in t) for t in termcoding)
    while True:
        maybe_termcoding = simplify_one(termcoding)
        if maybe_termcoding is None:
            return termcoding # We're done.
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

def categorical_coding(terms):
    # It is assumed that each element of `terms` describes an
    # interaction between zero or more categorical factors.
    decomposed = [decompose(t) for t in terms]
    non_redundant = [[t for t in term if not t in previous]
                     for term, previous in zip(decomposed, all_previous(decomposed))]
    return join(simplify(t) for t in non_redundant)


def partition(pred, iterable):
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)

# Generates a description of how the given terms ought to be coded
# into a design matrix.
def coding(terms, metadata):
    assert type(terms) == OrderedSet
    assert all(type(term) == Term for term in terms)
    assert type(metadata) == dict

    if not all(len(term.factors) < 2 or
               all(type(metadata[factor]) == Categorical for factor in term.factors) for term in terms):
        raise Exception('Interactions supported between categorical factors only.')

    def is_numeric_term(term):
        return len(term.factors) == 1 and not type(metadata[term.factors[0]]) == Categorical
    c_terms, n_terms = partition(is_numeric_term, terms)

    c_terms = sorted(c_terms, key=lambda term: len(term.factors))

    # It's possible to figure out c_terms and n_terms using the new
    # logic, but doing so clunky.
    # groups = partition_terms(terms, metadata)
    # if len(groups) == 0:
    #     c_terms = n_terms = []
    # elif groups[0][0] == OrderedSet():
    #     c_terms = sort_terms(groups[0][1])
    #     n_terms = [g[1][0] for g in groups[1:]]
    # else:
    #     c_terms = []
    #     n_terms = [g[1][0] for g in groups]

    def coded_factor_to_coding(tup):
        assert type(tup) == tuple
        # This is guaranteed by the check made on `terms` when
        # entering `coding`.
        assert all(type(cf) == CodedFactor and cf.factor in metadata for cf in tup)
        if len(tup) == 0:
            return InterceptC()
        else:
            # If there is more than one CodedFactor in tup, then this
            # represents the interaction between one or more
            # categorical facts. The coding of this is described by
            # `InteractionC`.
            return InteractionC([CategoricalC(metadata[cf.factor], cf.reduced) for cf in tup])

    c_coded = [coded_factor_to_coding(tup) for tup in categorical_coding(c_terms)]
    n_coded = [NumericC(term.factors[0]) for term in n_terms]
    # Following Patsy numeric terms come after categorical terms. This
    # is not what happens in R.
    return c_coded + n_coded



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
        if not k in out:
            out[k] = []
        out[k].append(v)
    return out


# Partition terms by the numeric factors they contain, and sort the
# resulting groups.
def partition_terms(terms, metadata):
    assert type(terms) == OrderedSet
    assert type(metadata) == dict

    def numeric_factors(term):
        factors = [f for f in term.factors if is_numeric_col(metadata[f])]
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
    return list(first) + list(rest)

# Terms with in a group are ordered by their order, i.e. the number of
# factors they contain.
def sort_terms(terms):
    assert type(terms) == list
    assert all(type(term) == Term for term in terms)
    return sorted(terms, key=lambda term: len(term.factors))


def width(coding):
    return sum(widthC(c) for c in coding)

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

# TODO: This could be generalised to work with a wider range of data
# representation. Rather than expecting a pandas dataframe, it could
# take a dictionary that gives access to the columns (iterables full
# of floats, ints, or level values?) and the existing dataframe
# metadata structure to describe the types of the columns, etc.
def designmatrix(terms, df):
    assert type(terms) == OrderedSet
    N = len(df)
    def dispatch(code):
        if type(code) == InterceptC:
            return [np.ones(N)]
        elif type(code) == InteractionC:
            return codeinteraction([df[c.factor.name] for c in code.codes],
                                   [c.reduced for c in code.codes])
        elif type(code) == NumericC:
            return codenumeric(df[code.name])
        else:
            raise Exception('Unknown coding type.')
    metadata = make_metadata_lookup(dfmetadata(df))
    coding_desc = coding(terms, metadata)
    coded_cols = join([dispatch(c) for c in coding_desc])
    X = np.stack([col2numpy(col) for col in coded_cols], axis=1) if coded_cols else np.empty((N, 0))
    assert X.shape == (N, width(coding_desc))
    if X.shape[1] > 0 and np.linalg.matrix_rank(X) != X.shape[1]:
        print('WARNING: Design matrix may not be full rank.')
    return X

# --------------------------------------------------

# TODO: Rename the "design matrix metadata" bits.

# It's a misnomer to call this design matrix metadata, a name I've
# long found confusing. I think a better name would pre/proto model
# (desc) or similar. That's because this an intermediate step towards
# a full `ModelDesc`. (`DesignMeta` is pretty much just a stripped
# down `ModelDesc`.) We start with a formula and some (meta)data, and
# from that we build one of these (pre/proto models). At this stage we
# know how the data will be coded, and therefore know what coefs
# appear in the model, but we don't yet have priors specified.

# Serving as the basis for prior specification is the only purpose of
# this structure -- once priors are specifed, the formula, prior tree
# and (meta)data carry enough information to build the `ModelDesc`. (I
# originally imagined this would be used in e.g. `marginals` but
# `ModelDesc` plays that role.)

# It might be possible to extend this structure to simplify some key
# functions. For example, `default_prior` currently takes formula,
# design matrix meta data and family argument, but if this pre/proto
# model were fleshed out with a little more information, it alone
# might be sufficient to build the default prior. Similarly,
# `build_model` requires a bunch or arguments, but one might expect
# that that just a pre/proto model and the prior tree would be
# sufficient to build a `ModelDesc`. (This function would then clearly
# be taking one model description to a second, richer, description.)

def numeric_metadata(code):
    return [code.name]

def interaction_metadata(code):
    assert type(code) == InteractionC
    assert all(type(c) == CategoricalC for c in code.codes)
    def coded_levels(c):
        return c.factor.levels[1:] if c.reduced else c.factor.levels
    interactions = product([tuple((c.factor.name, l)
                                  for l in coded_levels(c)) for c in code.codes])
    return [':'.join('{}[{}]'.format(factor, val)
                     for factor, val in interaction)
            for interaction in interactions]


def designmatrix_metadata(terms, metadata):
    assert type(terms) == OrderedSet
    def dispatch(code):
        if type(code) == InterceptC:
            return ['intercept']
        elif type(code) == InteractionC:
            return interaction_metadata(code)
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
    assert type(formula) == Formula
    assert type(metadata) == dict
    assert set(allfactors(formula)).issubset(set(metadata.keys()))
    p = PopulationMeta(designmatrix_metadata(formula.terms, metadata))
    gs = [GroupMeta(group.column, designmatrix_metadata(group.terms, metadata))
          for group in formula.groups]
    return DesignMeta(p, gs)

# --------------------------------------------------

def lookupvector(column, df):
    assert type(column) == str
    assert type(df) == pd.DataFrame
    assert column in df
    assert is_categorical_dtype(df[column])
    return df[column].cat.codes.to_numpy(np.int64)

def responsevector(column, df):
    assert type(column) == str
    assert type(df) == pd.DataFrame
    assert column in df
    dfcol = df[column]
    if is_float_dtype(dfcol) or is_integer_dtype(dfcol):
        coded = codenumeric(dfcol)
    elif is_categorical_dtype(dfcol) and len(dfcol.cat.categories) == 2:
        # TODO: How does a user know how this was coded? For design
        # matrices this is revealed by the column names in the design
        # metadata, but we don't have the here.
        coded = codefactor(dfcol, reduced=True)
    else:
        raise Exception('Don\'t know how to code a response of this type.')
    assert len(coded) == 1
    return col2numpy(coded[0])

def predictors(formula, df):
    assert type(formula) == Formula
    assert type(df) == pd.DataFrame
    data = {}
    data['X'] = designmatrix(formula.terms, df)
    for i, group in enumerate(formula.groups):
        data['Z_{}'.format(i)] = designmatrix(group.terms, df)
        data['J_{}'.format(i)] = lookupvector(group.column, df)
    return data

def makedata(formula, df):
    return dict(predictors(formula, df),
                y_obs=responsevector(formula.response, df))
