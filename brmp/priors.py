from collections import namedtuple, defaultdict
from pprint import pprint as pp

import pandas as pd

from pyro.contrib.brm.utils import join
from pyro.contrib.brm.formula import Formula
from pyro.contrib.brm.design import DesignMeta, PopulationMeta, GroupMeta
from pyro.contrib.brm.family import getfamily, Family, nonlocparams, Type

# `is_param` indicates whether a node corresponds to a parameter in
# the model. (Nodes without this flag set exist only to add structure
# to the parameters.) This infomation is used when extracting
# information from the tree.
Node = namedtuple('Node', 'name prior_edit is_param checks children')

def leaf(name, prior_edit=None, checks=[]):
    return Node(name, prior_edit, True, checks, [])

# TODO: This currently requires `parameters` to be a list of floats.
# (Actually, the family gives us more info than this, e.g. that a
# parameter is constrained to take only positive reals.) This ought to
# be checked.

# TODO: Can also check that the number of args matches the number of
# params. (Having done this, is would be safe to remove the assertion
# that LKJ has a single param that's currently in `codgen.py`.)

Prior = namedtuple('Prior', 'family arguments')

# e.g. Prior('Normal', [0., 1.])
def prior(family_name, args):
    return Prior(getfamily(family_name), args)

RESPONSE_PRIORS = {
    'Normal': {
        'sigma': Prior(getfamily('HalfCauchy'), [3.])
    }
}

def get_response_prior(family, parameter):
    return RESPONSE_PRIORS[family][parameter]

# This is similar to brms `set_prior`. (e.g. `set_prior('<prior>',
# coef='x1')` is similar to `PriorEdit(['x1'], '<prior>)`.) By
# specifying paths (rather than class/group/coef) we're diverging from
# brms, but the hope is that a brms-like interface can be put in front
# of this.

PriorEdit = namedtuple('PriorEdit', 'path prior')

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

def default_prior(formula, design_metadata, family):
    assert type(formula) == Formula
    assert type(design_metadata) == DesignMeta
    assert type(family) == Family
    assert family.response is not None
    assert type(design_metadata.population) == PopulationMeta
    assert type(design_metadata.groups) == list
    assert all(type(gm) == GroupMeta for gm in design_metadata.groups)
    # It's assumed that `formula` and `design_metadata` are
    # consistent. Something like, there exists dataframe metadata
    # `metadata` s.t.:
    # `design_metadata = designmatrices_metadata(formula, metadata)`
    # This sanity checks the these two agree about which groups are present.
    assert all(meta.name == group.column
               for (meta, group)
               in zip(design_metadata.groups, formula.groups))
    b_children = [leaf(name) for name in design_metadata.population.coefs]
    cor_children = [leaf(group.column) for group in formula.groups if group.corr and len(group.gterms) > 1]
    sd_children = [Node(gm.name, None, False, [], [leaf(name) for name in gm.coefs]) for gm in design_metadata.groups]
    resp_children = [leaf(p.name,
                          PriorEdit(('resp', p.name), get_response_prior(family.name, p.name)),
                          [chk_support(p.type)])
                     for p in nonlocparams(family)]
    return Node('root', None, False, [], [
        Node('b',    PriorEdit(('b',),   prior('Cauchy', [0., 1.])), False, [chk_support(Type.real)],     b_children),
        Node('sd',   PriorEdit(('sd',),  prior('HalfCauchy', [3.])), False, [chk_support(Type.pos_real)], sd_children),
        Node('cor',  PriorEdit(('cor',), prior('LKJ', [1.])),        False, [chk_lkj],                    cor_children),
        Node('resp', None,                                           False, [],                           resp_children)])

# TODO: This ought to warn/error when an element of `priors` has a
# path that doesn't correspond to a node in the tree.

def customize_prior(tree, prior_edits):
    assert type(tree) == Node
    assert type(prior_edits) == list
    assert all(type(p) == PriorEdit for p in prior_edits)
    for prior_edit in prior_edits:
        tree = edit(tree, prior_edit.path,
                    lambda n: Node(n.name, prior_edit, n.is_param, n.checks, n.children))
    return tree

# It's important that trees maintain the order of their children, so
# that coefficients in the prior tree continue to line up with columns
# in the design matrix.
def build_prior_tree(formula, design_metadata, family, prior_edits, chk=True):
    tree = fill(customize_prior(default_prior(formula, design_metadata, family), prior_edits))
    if chk:
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

    def __call__(self, prior):
        assert type(prior) == Prior
        return self.predicate(prior)

    def __repr__(self):
        return 'Chk("{}")'.format(self.name)

def chk(name):
    def decorate(predicate):
        return Chk(predicate, name)
    return decorate

def chk_support(typ):
    def pred(prior):
        return prior.family.support == typ
    return Chk(pred, 'has support of {}'.format(typ.name))

@chk('is LKJ')
def chk_lkj(prior):
    return prior.family.name == 'LKJ'

def check(tree):
    errors = defaultdict(lambda: defaultdict(list))
    for (node, path) in leaves(tree):
        for chk in node.checks:
            if not chk(node.prior_edit.prior):
                errors[node.prior_edit.path][chk].append(path)
    return errors

# TODO: There's info in `errors` which we're not making use of here.
def format_errors(errors):
    paths = ', '.join('"{}"'.format('/'.join(path))
                      for path in errors.keys())
    return 'Invalid prior specified at {}.'.format(paths)

def main():
    from pyro.contrib.brm.formula import parse
    from pyro.contrib.brm.design import designmatrices_metadata, make_metadata_lookup

    formula = parse('y ~ 1 + x1 + x2 + (1 || grp1) + (1 + z | grp2) + (1 | grp3)')
    design_metadata = designmatrices_metadata(
        formula,
        make_metadata_lookup([]))
    prior_edits = [
        PriorEdit(('b',), 'b'),
        PriorEdit(('sd',), 'a'),
        PriorEdit(('sd', 'grp2',), 'c'),
        PriorEdit(('sd', 'grp2', 'z',), 'd'),
        PriorEdit(('cor',), 'e'),
        PriorEdit(('cor', 'grp3',), 'f'),
        PriorEdit(('resp', 'sigma',), 'g'),
    ]

    tree = build_prior_tree(formula, design_metadata, getfamily('Normal'), prior_edits, chk=False)
    pp([('/'.join(path), node.prior_edit.prior) for node, path in leaves(tree)])
    # [('b/intercept', 'b'),
    #  ('b/x1', 'b'),
    #  ('b/x2', 'b'),
    #  ('sd/grp1/intercept', 'a'),
    #  ('sd/grp2/intercept', 'c'),
    #  ('sd/grp2/z', 'd'),
    #  ('sd/grp3/intercept', 'a'),
    #  ('cor/grp2', 'e'),
    #  ('resp/sigma', 'g')]

    tree = build_prior_tree(formula, design_metadata, getfamily('Normal'), [
        # This edit will fail the +ve support check at all the grand
        # children of the sd node.
        PriorEdit(('sd',), prior('Normal', [0., 1.])),
    ], chk=False)
    pp(dict((k, dict(v)) for k, v in check(tree).items()))
    # {('sd',): {Chk("has +ve support"): [('sd', 'grp1', 'intercept'),
    #                                     ('sd', 'grp2', 'intercept'),
    #                                     ('sd', 'grp2', 'z'),
    #                                     ('sd', 'grp3', 'intercept')]}}

if __name__ == '__main__':
    main()
