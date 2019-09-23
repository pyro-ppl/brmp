from collections import namedtuple

from .formula import Formula, allfactors
from .design import Metadata, coef_names, RealValued, Categorical, Integral
from .family import Family, Type, nonlocparams, support_depends_on_args, family_repr

def family_matches_response(formula, metadata, family):
    assert type(formula) == Formula
    assert type(metadata) == Metadata
    assert type(family) == Family
    # I don't think there is any way for this not to hold with the
    # present system. However, it /could/ arise if it were possible to
    # put a prior over e.g. the `num_trials` parameter of Binomial,
    # for example. Because this holds we know we can safely
    # call`family.support` with zero args below.
    assert not support_depends_on_args(family)
    factor = metadata.column(formula.response)
    if type(family.support()) == Type['Real']:
        return type(factor) == RealValued
    elif type(family.support()) == Type['Boolean']:
        if type(factor) == Categorical:
            return len(factor.levels) == 2
        elif type(factor) == Integral:
            return factor.min == 0 and factor.max == 1
        else:
            return False
    elif (type(family.support()) == Type['IntegerRange']):
        factor = metadata.column(formula.response)
        return (type(factor) == Integral and
                (family.support().lb is None or factor.min >= family.support().lb) and
                (family.support().ub is None or factor.max <= family.support().ub))
    elif type(family.support()) == Type['UnitInterval']:
        return (type(factor) == RealValued and
                factor.min is not None and
                factor.max is not None and
                factor.min >= 0. and
                factor.max <= 1.)
    else:
        return False

def check_family_matches_response(formula, metadata, family):
    assert type(metadata) == Metadata
    if not family_matches_response(formula, metadata, family):
        # TODO: This could be more informative. e.g. If choosing
        # Bernoulli fails, is the problem that the response is
        # numeric, or that it has more than two levels?
        error = 'The response distribution "{}" is not compatible with the type of the response column "{}".'
        raise Exception(error.format(family_repr(family), formula.response))

# `ModelDescPre` is an intermediate step towards a full `ModelDesc`.
# At this stage we know how the data will be coded, and therefore know
# what coefs appear in the model, but we don't yet have priors
# specified.

# This structure serves as the basis for prior specification. Once the
# prior tree is built, it is combined with this `ModelDescPre` to
# produce the final `ModelDesc`.

ModelDescPre = namedtuple('ModelDescPre', 'population groups response')
PopulationPre = namedtuple('PopulationPre', 'coefs')
GroupPre = namedtuple('GroupPre', 'columns levels coefs corr')
ResponsePre = namedtuple('ResponsePre', 'family nonlocparams')

def build_model_pre(formula, metadata, family, custom_code_lengths):
    assert type(formula) == Formula
    assert type(metadata) == Metadata
    assert type(family) == Family
    assert set(allfactors(formula)).issubset(set(col.name for col in metadata.columns))

    check_family_matches_response(formula, metadata, family)

    p = PopulationPre(coef_names(formula.terms, metadata, custom_code_lengths))

    gs = []
    for group in formula.groups:
        assert all(type(metadata.column(col)) == Categorical for col in group.columns), 'grouping columns must be a factor'
        coefs = coef_names(group.terms, metadata, custom_code_lengths)
        g = GroupPre(group.columns, metadata.levels(group.columns), coefs, group.corr and len(coefs) > 1)
        gs.append(g)

    response = ResponsePre(family, nonlocparams(family))
    return ModelDescPre(p, gs, response)
