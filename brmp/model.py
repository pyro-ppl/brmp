from .formula import Formula
from .family import Family, Type

# This could be the place where an abstract model description is
# built. (Having performed any high-level checks required to ensure
# that the requested model is sane.)

def family_matches_response(formula, metadata, family):
    assert type(formula) == Formula
    assert type(metadata) == dict
    assert type(family) == Family
    if not formula.response in metadata:
        # Response column is numeric.
        return family.support == Type.real
    else:
        # Response column is a factor.
        factor = metadata[formula.response]
        if len(factor.levels) == 2:
            return family.support == Type.boolean
        else:
            return False

def check_family_matches_response(formula, metadata, family):
    if not family_matches_response(formula, metadata, family):
        # TODO: This could be more informative. e.g. If choosing
        # Bernoulli fails, is the problem that the response is
        # numeric, or that it has more than two levels?
        error = 'The "{}" family is not compatible with the type of response column "{}".'
        raise Exception(error.format(family.name, formula.response))
