from collections import namedtuple, defaultdict
from pprint import pprint as pp

import pandas as pd

from pyro.contrib.brm.utils import join
from pyro.contrib.brm.formula import Formula
from pyro.contrib.brm.model_pre import ModelDescPre, PopulationPre, GroupPre
from pyro.contrib.brm.family import Cauchy, HalfCauchy, LKJ, Family, nonlocparams, Type, fully_applied

# `is_param` indicates whether a node corresponds to a parameter in
# the model. (Nodes without this flag set exist only to add structure
# to the parameters.) This infomation is used when extracting
# information from the tree.
Node = namedtuple('Node', 'name prior_edit is_param checks children')

def leaf(name, prior_edit=None, checks=[]):
    return Node(name, prior_edit, True, checks, [])

RESPONSE_PRIORS = {
    'Normal': {
        'sigma': HalfCauchy(3.)
    },
}

def get_response_prior(family, parameter):
    if family in RESPONSE_PRIORS:
        return RESPONSE_PRIORS[family][parameter]

# This is similar to brms `set_prior`. (e.g. `set_prior('<prior>',
# coef='x1')` is similar to `Prior(['x1'], '<prior>)`.) By specifying
# paths (rather than class/group/coef) we're diverging from brms, but
# the hope is that a brms-like interface can be put in front of this.

Prior = namedtuple('Prior', 'path prior')

def walk(node, path):
    assert type(node) == Node
    assert type(path) == tuple
    if len(path) == 0:
        return [node]
    else:
        name = path[0]
        selected_node = next((n for n in node.children if n.name == name), None)
        if selected_node is None:
            raise ValueError('Invalid path')
        return [node] + walk(selected_node, path[1:])

def select(node, path):
    return walk(node, path)[-1]

def tryselect(node, path, default=None):
    try:
        return select(node, path)
    except ValueError:
        return default

def edit(node, path, f):
    assert type(node) == Node
    assert type(path) == tuple
    if len(path) == 0:
        # We're at the node to be edited. (i.e. Empty path picks out
        # the root node.)
        newnode = f(node)
        assert type(newnode) == Node
        return newnode
    else:
        # Recursively edit the appropriate child. (Or children, if
        # names are duplicated.)
        name = path[0]
        assert any(n.name == name for n in node.children), 'Node "{}" not found.'.format(name)
        children = [edit(n, path[1:], f) if n.name == name else n
                    for n in node.children]
        return Node(node.name, node.prior_edit, node.is_param, node.checks, children)

# TODO: Match default priors used by brms. (An improper uniform is
# used for `b`. A Half Student-t here is used for priors on standard
# deviations, with its scale derived from the data.)

# TODO: It might be a good idea to build the tree with checks but no
# priors, and then add the priors using in the same way as user edits
# are applied, in order to ensure that the default meet the
# contraints. Or, perhaps a more convenient way of achieving the same
# thing is to make an separate pass over the entire default tree once
# built, and assert its consistency.

def default_prior(model_desc_pre, family):
    assert type(model_desc_pre) == ModelDescPre
    assert type(family) == Family
    assert family.link is not None
    assert type(model_desc_pre.population) == PopulationPre
    assert type(model_desc_pre.groups) == list
    assert all(type(gm) == GroupPre for gm in model_desc_pre.groups)
    b_children = [leaf(name) for name in model_desc_pre.population.coefs]
    cor_children = [leaf(cols2str(group.columns)) for group in model_desc_pre.groups if group.corr]
    sd_children = [Node(cols2str(gm.columns), None, False, [], [leaf(name) for name in gm.coefs]) for gm in model_desc_pre.groups]

    def mk_resp_prior_edit(param_name, family_name):
        prior = get_response_prior(family_name, param_name)
        if prior is not None:
            return Prior(('resp', param_name), prior)

    resp_children = [leaf(p.name, mk_resp_prior_edit(p.name, family.name), [chk_support(p.type)])
                     for p in nonlocparams(family)]
    return Node('root', None, False, [], [
        Node('b',    Prior(('b',),   Cauchy(0., 1.)), False, [chk_support(Type['Real']())],    b_children),
        Node('sd',   Prior(('sd',),  HalfCauchy(3.)), False, [chk_support(Type['PosReal']())], sd_children),
        Node('cor',  Prior(('cor',), LKJ(1.)),        False, [chk_lkj],                        cor_children),
        Node('resp', None,                            False, [],                               resp_children)])

def cols2str(cols):
    return ':'.join(cols)


def customize_prior(tree, priors):
    assert type(tree) == Node
    assert type(priors) == list
    assert all(type(p) == Prior for p in priors)
    for prior_edit in priors:
        # TODO: It probably makes sense to move this to the
        # constructor of Prior, once such a thing exists.
        if not fully_applied(prior_edit.prior):
            raise Exception('Distribution arguments missing from prior "{}"'.format(prior_edit.prior.name))
        tree = edit(tree, prior_edit.path,
                    lambda n: Node(n.name, prior_edit, n.is_param, n.checks, n.children))
    return tree

# It's important that trees maintain the order of their children, so
# that coefficients in the prior tree continue to line up with columns
# in the design matrix.
def build_prior_tree(model_desc_pre, family, priors, chk=True):
    tree = fill(customize_prior(default_prior(model_desc_pre, family), priors))
    if chk:
        # TODO: I might consider delaying this check (that all
        # parameters have priors) until just before code generation
        # happens. This could allow an under-specified model to be
        # pretty-printed, which might make it easier for users to see
        # what's going on. (Once `brm` returns a model rather than
        # running inference.) Doing so would require the `ModelDesc`
        # data structure and pretty printing code to handle missing
        # priors. (Does something similar apply to the response/family
        # compatibility checks currently in model.py?)
        missing_prior_paths = leaves_without_prior(tree)
        if len(missing_prior_paths) > 0:
            paths = ', '.join('"{}"'.format('/'.join(path)) for path in missing_prior_paths)
            raise Exception('Prior missing from {}.'.format(paths))
        errors = check(tree)
        if errors:
            raise Exception(format_errors(errors))
    return tree

# `fill` populates the `prior_edit` and `checks` properties of all
# nodes in a tree. Each node uses its own `prior_edit` value if set,
# otherwise the first `prior_edit` value encountered when walking up
# the tree from the node is used. The final values of `checks` comes
# from concatenating all of the lists of checks encountered when
# walking from the node to the root. (This is the behaviour, not the
# implementation.)
def fill(node, default=None, upstream_checks=[]):
    prior = node.prior_edit if not node.prior_edit is None else default
    checks = upstream_checks + node.checks
    return Node(node.name, prior, node.is_param, checks, [fill(n, prior, checks) for n in node.children])

def leaves(node, path=[]):
    this = [(node, path)] if node.is_param else []
    rest = join(leaves(n, path + [n.name]) for n in node.children)
    return this + rest

# Sanity checks

class Chk():
    def __init__(self, predicate, name):
        self.predicate = predicate
        self.name = name

    def __call__(self, node):
        assert type(node) == Node
        if node.prior_edit is None:
            # There is no prior to check.
            return True
        else:
            return self.predicate(node.prior_edit.prior)

    def __repr__(self):
        return 'Chk("{}")'.format(self.name)

def chk(name):
    def decorate(predicate):
        return Chk(predicate, name)
    return decorate

def chk_support(typ):
    # TODO: This could probably be relaxed to only require that the
    # support of the prior is a subset of type of the parameter.
    # (However this is easier and good enough for now.)
    def pred(prior):
        return prior.support() == typ
    return Chk(pred, 'has support of {}'.format(typ))

@chk('is LKJ')
def chk_lkj(prior):
    return prior.name == 'LKJ'

def check(tree):
    errors = defaultdict(lambda: defaultdict(list))
    for (node, path) in leaves(tree):
        for chk in node.checks:
            if not chk(node):
                # This holds because checks can only fail when a node
                # has a `prior_edit`.
                assert node.prior_edit is not None
                errors[node.prior_edit.path][chk].append(path)
    return errors

# TODO: There's info in `errors` which we're not making use of here.
def format_errors(errors):
    paths = ', '.join('"{}"'.format('/'.join(path))
                      for path in errors.keys())
    return 'Invalid prior specified at {}.'.format(paths)

def leaves_without_prior(tree):
    return [path for (node, path) in leaves(tree) if node.prior_edit is None]
