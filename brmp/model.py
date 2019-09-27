from collections import namedtuple

from brmp.utils import unzip
from brmp.family import Family, family_repr
from brmp.priors import select, Node, cols2str
from brmp.model_pre import ModelDescPre

# Abstract model description.
ModelDesc = namedtuple('ModelDesc', 'population groups response')
Population = namedtuple('Population', 'coefs priors')
Group = namedtuple('Group', 'columns levels coefs sd_priors corr_prior')
Response = namedtuple('Response', 'family nonlocparams priors')

# Add information from the prior tree to the pre-model description.
def build_model(model_desc_pre, prior_tree):
    assert type(model_desc_pre) == ModelDescPre
    assert type(prior_tree) == Node

    node = select(prior_tree, ('b',))
    b_coefs, b_priors = unzip([(n.name, n.prior_edit.prior) for n in node.children])
    # Sanity check. Assert that the coef names pulled from the tree
    # match those in the pre-model.
    assert model_desc_pre.population.coefs == list(b_coefs)
    population = Population(model_desc_pre.population.coefs, b_priors)

    groups = []
    for group in model_desc_pre.groups:

        grp_node_name = cols2str(group.columns)

        # If the pre-model indicates there is a corr prior, look it up
        # in the tree.
        corr_prior = select(prior_tree, ('cor', grp_node_name)).prior_edit.prior if group.corr else None
        assert corr_prior is None or type(corr_prior) == Family

        sd_coefs, sd_priors = unzip([(n.name, n.prior_edit.prior) for n in select(prior_tree, ('sd', grp_node_name)).children])

        # Sanity check. Assert that the coef names pulled from the
        # tree match those in the pre-model.
        assert group.coefs == list(sd_coefs)

        groups.append(Group(group.columns, group.levels, group.coefs, sd_priors, corr_prior))

    nl_priors = [n.prior_edit.prior for n in select(prior_tree, ('resp',)).children]
    response = Response(model_desc_pre.response.family,
                        model_desc_pre.response.nonlocparams,
                        nl_priors)
    # Sanity check. Ensure we have the correct number of priors.
    assert len(response.nonlocparams) == len(response.priors)

    return ModelDesc(population, groups, response)


def model_repr(model):
    assert type(model) == ModelDesc
    out = []
    def write(s):
        out.append(s)
    write('=' * 40)
    write('Population')
    write('-' * 40)
    write('Coef Priors:')
    for (coef, prior) in zip(model.population.coefs, model.population.priors):
        write('{:<15} | {}'.format(coef, family_repr(prior)))
    for i, group in enumerate(model.groups):
        write('=' * 40)
        write('Group {}'.format(i))
        write('-' * 40)
        write('Factors: {}\nNum Levels: {}'.format(', '.join(group.columns), len(group.levels)))
        write('Corr. Prior: {}'.format(None if group.corr_prior is None else family_repr(group.corr_prior)))
        write('S.D. Priors:')
        for (coef, sd_prior) in zip(group.coefs, group.sd_priors):
            write('{:<15} | {}'.format(coef, family_repr(sd_prior)))
    write('=' * 40)
    write('Response')
    write('-' * 40)
    write('Family: {}'.format(family_repr(model.response.family)))
    write('Link:')
    write('  Parameter: {}'.format(model.response.family.link.param))
    write('  Function:  {}'.format(model.response.family.link.fn.name))
    write('Priors:')
    for (param, prior) in zip(model.response.nonlocparams, model.response.priors):
        write('{:<15} | {}'.format(param.name, family_repr(prior)))
    write('=' * 40)
    return '\n'.join(out)


Parameter = namedtuple('Parameter', ['name', 'shape'])

def parameter_names(model):
    return [parameter.name for parameter in parameters(model)]

# This describes the set of parameters implied by a particular model.
# Any backend is expected to produce models the make this set of
# parameters available (with the described shapes) via its `get_param`
# function. (See fit.py.)
def parameters(model):
    assert type(model) == ModelDesc
    return ([Parameter('b', (len(model.population.coefs),))] +
            [Parameter('r_{}'.format(i), (len(group.levels), len(group.coefs)))
             for i, group in enumerate(model.groups)] +
            [Parameter('sd_{}'.format(i), (len(group.coefs),))
             for i, group in enumerate(model.groups)] +
            [Parameter('L_{}'.format(i), (len(group.coefs), len(group.coefs)))
             for i, group in enumerate(model.groups) if not group.corr_prior is None] +
            [Parameter(param.name, (1,)) for param in model.response.nonlocparams])

# [ (scalar_param_name, (param_name, (ix0, ix1, ...)), ... ]

# (ix0, ix1, ...) is an index into that the parameter (picked out by
# `param_name`) once put through `to_numpy`. This is a tuple because
# parameters are not necessarily vectors.

def scalar_parameter_map(model):
    assert type(model) == ModelDesc
    out = [('b_{}'.format(coef), ('b', (i,)))
           for i, coef in enumerate(model.population.coefs)]
    for ix, group in enumerate(model.groups):
        out.extend([('sd_{}__{}'.format(cols2str(group.columns), coef), ('sd_{}'.format(ix), (i,)))
                    for i, coef in enumerate(group.coefs)])
        out.extend([('r_{}[{},{}]'.format(cols2str(group.columns), '_'.join(level), coef), ('r_{}'.format(ix), (i, j)))
                    for j, coef in enumerate(group.coefs)
                    for i, level in enumerate(group.levels)])
    for param in model.response.nonlocparams:
        out.append((param.name, (param.name, (0,))))
    return out

def scalar_parameter_names(model):
    return [name for (name, _) in scalar_parameter_map(model)]
