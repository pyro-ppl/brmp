from collections import namedtuple

from pyro.contrib.brm.utils import unzip
from .formula import Formula
from .family import Family, Type, nonlocparams
from .priors import select, tryselect, Prior, Node

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


Model = namedtuple('Model', 'population groups response')
Population = namedtuple('Population', 'coefs priors')
Group = namedtuple('Group', 'factor coefs sd_priors corr_prior')
Response = namedtuple('Response', 'family nonlocparams priors')


def build_model(formula, prior_tree, family, dfmetadata):
    assert type(formula) == Formula
    assert type(prior_tree) == Node
    assert type(family) == Family
    assert type(dfmetadata) == dict

    # TODO: `formula` is only used in order to perform the following
    # check. Internally, the information about the response column
    # name is used to perform the check. So, does it make sense for
    # `build_model` to take only the response column as argument?
    # Alternatively, perhaps it makes sense for this information could
    # be incorporated in design meta, or otherwise included in one of
    # the args. already received.
    check_family_matches_response(formula, dfmetadata, family)

    # Population-level
    node = select(prior_tree, ('b',))
    b_coefs, b_priors = unzip([(n.name, n.prior_edit.prior) for n in node.children])
    population = Population(b_coefs, b_priors)
    # Assert invariant.
    assert len(population.coefs) == len(population.priors)

    # Groups
    groups = []

    for node in select(prior_tree, ('sd',)).children:

        assert node.name in dfmetadata, 'group column must be a factor'

        sd_coefs, sd_priors = unzip([(n.name, n.prior_edit.prior) for n in node.children])

        corr_node = tryselect(prior_tree, ('cor', node.name))
        corr_prior = None if corr_node is None else corr_node.prior_edit.prior

        group = Group(dfmetadata[node.name], sd_coefs, sd_priors, corr_prior)
        # Assert invariants.
        assert len(group.coefs) == len(group.sd_priors)
        assert group.corr_prior is None or type(group.corr_prior) == Prior
        groups.append(group)

    nl_params = nonlocparams(family)
    nl_priors = [n.prior_edit.prior for n in select(prior_tree, ('resp',)).children]
    response = Response(family, nl_params, nl_priors)
    # Assert invariants.
    assert len(response.nonlocparams) == len(response.priors)

    return Model(population, groups, response)


def model_repr(model):
    assert type(model) == Model
    out = []
    def write(s):
        out.append(s)
    # TODO: Move to a `Prior` class?
    def prior_repr(prior):
        return '{}({})'.format(prior.family.name, ', '.join([str(arg) for arg in prior.arguments]))
    write('=' * 40)
    write('Population')
    write('-' * 40)
    write('Coef Priors:')
    for (coef, prior) in zip(model.population.coefs, model.population.priors):
        write('{:<15} | {}'.format(coef, prior_repr(prior)))
    for i, group in enumerate(model.groups):
        write('=' * 40)
        write('Group {}'.format(i))
        write('-' * 40)
        write('Factor: {}\nLevels: {}'.format(group.factor.name, group.factor.levels))
        write('Corr. Prior: {}'.format(None if group.corr_prior is None else prior_repr(group.corr_prior)))
        write('S.D. Priors:')
        for (coef, sd_prior) in zip(group.coefs, group.sd_priors):
            write('{:<15} | {}'.format(coef, prior_repr(sd_prior)))
    write('=' * 40)
    write('Response')
    write('-' * 40)
    write('Family: {}'.format(model.response.family.name))
    write('Link:')
    write('  Parameter: {}'.format(model.response.family.response.param))
    write('  Function:  {}'.format(model.response.family.response.linkfn.name))
    write('Priors:')
    for (param, prior) in zip(model.response.nonlocparams, model.response.priors):
        write('{:<15} | {}'.format(param.name, prior_repr(prior)))
    write('=' * 40)
    return '\n'.join(out)


# TODO: The choice to use 1-based indexing is made in many places.
# Consolidate?

# TODO: Are these really best called parameters, or is there something
# better?

Parameter = namedtuple('Parameter', ['name', 'shape'])

def parameter_names(model):
    return [parameter.name for parameter in parameters(model)]

# This describes the set of parameters implied by a particular model.
# Any backend is expected to produce models the make this set of
# parameters available (with the described shapes) via its `get_param`
# function. (See fit.py.)
def parameters(model):
    return ([Parameter('b', (len(model.population.coefs),))] +
            [Parameter('r_{}'.format(i+1), (len(group.factor.levels), len(group.coefs)))
             for i, group in enumerate(model.groups)] +
            [Parameter('sd_{}'.format(i+1), (len(group.coefs),))
             for i, group in enumerate(model.groups)] +
            [Parameter('L_{}'.format(i+1), (len(group.coefs), len(group.coefs)))
             for i, group in enumerate(model.groups) if not group.corr_prior is None] +
            [Parameter(param.name, (1,)) for param in model.response.nonlocparams])
