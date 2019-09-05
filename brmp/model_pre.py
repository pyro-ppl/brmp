from collections import namedtuple

from .formula import Formula, allfactors
from .design import Metadata, coef_names
from .family import Family, nonlocparams

# `ModelDescPre` is an intermediate step towards a full `ModelDesc`.
# We start with a formula and some (meta)data, and from that we build
# one of these pre/proto models. At this stage we know how the data
# will be coded, and therefore know what coefs appear in the model,
# but we don't yet have priors specified.

# Serving as the basis for prior specification is the only purpose of
# this structure -- once priors are specifed, the formula, prior tree
# and (meta)data carry enough information to build the `ModelDesc`. (I
# originally imagined this would be used in e.g. `marginals` but
# `ModelDesc` plays that role.)

ModelDescPre = namedtuple('ModelDescPre', 'population groups response')
PopulationPre = namedtuple('PopulationPre', 'coefs')
GroupPre = namedtuple('GroupPre', 'columns levels coefs corr')
ResponsePre = namedtuple('ResponsePre', 'family nonlocparams')

def build_model_pre(formula, metadata, family):
    assert type(formula) == Formula
    assert type(metadata) == Metadata
    assert set(allfactors(formula)).issubset(set(col.name for col in metadata.columns))
    p = PopulationPre(coef_names(formula.terms, metadata))
    gs = [GroupPre(group.columns, metadata.levels(group.columns), coefs, group.corr and len(coefs) > 1)
          for group, coefs in ((group, coef_names(group.terms, metadata))
                               for group in formula.groups)]
    response = ResponsePre(family, nonlocparams(family))
    return ModelDescPre(p, gs, response)
