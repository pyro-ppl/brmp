from collections import namedtuple
from pprint import pprint as pp

import pandas as pd
import torch.distributions.constraints as constraints
import pyro.distributions as dists

from pyro.contrib.brm.utils import join
from pyro.contrib.brm.formula import Formula, parse
from pyro.contrib.brm.design import designmatrices_metadata, DesignMeta, PopulationMeta, GroupMeta, make_metadata_lookup
from pyro.contrib.brm.family import getfamily, Family, nonlocparams

Node = namedtuple('Node', 'name prior checks children')

def leaf(name):
    return Node(name, None, [], [])

# TODO: This currently requires `parameters` to be a list of floats.
# This ought to be checked.
Prior = namedtuple('Prior', 'family arguments')

# e.g. Prior('Normal', [0., 1.])
def prior(family_name, args):
    return Prior(getfamily(family_name), args)

RESPONSE_PRIORS = {
    'Normal': {
        'sigma': Prior(getfamily('HalfCauchy'), [3.])
    }
}

def check_response_priors_is_complete():
    # TODO: Ensure every family with a Response has priors for all
    # non-location args.
    pass

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
    assert type(path) == list
    if len(path) == 0:
        return [node]
    else:
        name = path[0]
        selected_node = next((n for n in node.children if n.name == name), None)
        if selected_node is None:
            raise Exception('Invalid path')
        return [node] + walk(selected_node, path[1:])

def select(node, path):
    return walk(node, path)[-1]

def edit(node, path, f):
    assert type(node) == Node
    assert type(path) == list
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
        return Node(node.name, node.prior, node.checks, children)

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
    cor_children = [Node(group.column, None, [], []) for group in formula.groups if group.corr]
    sd_children = [Node(gm.name, None, [], [leaf(name) for name in gm.coefs]) for gm in design_metadata.groups]
    # TODO: Consider adding a check that ensures the support of the
    # prior matches any constraint on the parameter. (Would require
    # families extending with additional info.)
    resp_children = [Node(p, get_response_prior(family.name, p), [], []) for p in nonlocparams(family)]
    return Node('root', None, [], [
        Node('b',    prior('Cauchy', [0., 1.]), [], b_children),
        Node('sd',   prior('HalfCauchy', [3.]), [chk_pos_support], sd_children),
        Node('cor',  prior('LKJ', [1.]),        [chk_lkj], cor_children),
        Node('resp', None,                      [], resp_children)])

# TODO: This ought to warn/error when an element of `priors` has a
# path that doesn't correspond to a node in the tree.

# TODO: We need to check (somehow, somewhere) that users specified
# priors have the correct support. e.g. R for population level
# effects, R+ for standard deviations.

def customize_prior(tree, prior_edits):
    assert type(tree) == Node
    assert type(prior_edits) == list
    assert all(type(p) == PriorEdit for p in prior_edits)
    for p in prior_edits:
        tree = edit(tree, p.path,
                    lambda n: Node(n.name, p.prior, n.checks, n.children))
    return tree

# It's important that trees maintain the order of their children,
# otherwise the output of `get_priors` will silently fail to line-up
# with the column ordering in the data.
def build_prior_tree(formula, design_metadata, family, prior_edits):
    return fill(customize_prior(default_prior(formula, design_metadata, family), prior_edits))


# `fill` populates the `prior` property of all nodes in a tree. Each
# node uses its own `prior` value if set, otherwise the first `prior`
# value encountered when walking up the tree from the node is used.
# (This is the behaviour, not the implementation.)
def fill(node, default=None):
    prior = node.prior if not node.prior is None else default
    return Node(node.name, prior, node.checks, [fill(n, prior) for n in node.children])


def leaves(node, path=[]):
    if len(node.children) == 0:
        # Return `node.prior` rather than the entire `node`, since the
        # other properties are recoverable from path and fact that
        # node is a leaf.
        return [(path, node.prior)]
    else:
        return join(leaves(n, path + [n.name]) for n in node.children)

# e.g.
# contig(list('abb')) == [('a', 1), ('b', 2)]
def contig(xs):
    assert type(xs) == list # Though really more general than this.
    assert all(x is not None for x in xs) # Since None used as initial value of `cur`.
    cur = None
    segments = []
    for i, x in enumerate(xs):
        if x == cur:
            segments[-1][1].append(i) # Extend segment.
        else:
            cur = x
            segments.append((cur, [i])) # New segment.
    # Post-process.
    segments = [(x, len(ix)) for (x, ix) in segments]
    return segments

# TODO: I'm missing opportunities to vectorise here. Adjacent segments
# that share a family and differ only in parameters can be handled
# with a single `sample` statement with suitable parameters.

def get_priors(formula, design_metadata, family, prior_edits, chk=True):
    assert type(formula) == Formula
    assert type(design_metadata) == DesignMeta
    assert type(family) == Family
    assert type(prior_edits) == list
    tree = build_prior_tree(formula, design_metadata, family, prior_edits)
    if chk:
        check_prior_edits(tree, prior_edits)
    def get(path):
        return contig([n.prior for n in select(tree, path).children])
    return dict(
        b=get(['b']),
        sd=dict((group_meta.name, get(['sd', group_meta.name]))
                for group_meta in design_metadata.groups),
        cor=dict((n.name, n.prior) for n in select(tree, ['cor']).children),
        resp=dict((n.name, n.prior) for n in select(tree, ['resp']).children))

# Sanity checks

def chk(error):
    def decorate(predicate):
        def f(prior):
            assert type(prior) == Prior
            retval = predicate(prior)
            assert type(retval) == bool
            return None if retval else error
        f.predicate = predicate
        return f
    return decorate

# TODO: Replace this Pyro specific check with something backend
# agnostic. (By extending the info we keep about families to include
# information about their support.)
@chk('A distribution with support on only the positive reals expected here.')
def chk_pos_support(prior):
    dist = dists.__getattribute__(prior.family.name)
    return dist.support == constraints.positive

@chk('Only the LKJ(...) family is supported here.')
def chk_lkj(prior):
    return prior.family.name == 'LKJ'

# TODO: We could have a further check that attempt to instantiate the
# distribution as a way of validating parameters? LKJ would again
# require special casing, since that ends up as `LKJCorrCholesky` in
# code.

def getchecks(node, path):
    return join(n.checks for n in walk(node, path))

def check_prior_edit(tree, edit):
    assert type(tree) == Node
    assert type(edit) == PriorEdit
    for chk in getchecks(tree, edit.path):
        maybe_error = chk(edit.prior)
        if not maybe_error is None:
            return maybe_error

def check_prior_edits(tree, edits):
    assert type(tree) == Node
    assert type(edits) == list
    for edit in edits:
        maybe_error = check_prior_edit(tree, edit)
        if not maybe_error is None:
            raise Exception(maybe_error)

def main():
    formula = parse('y ~ 1 + x1 + x2 + (1 || grp1) + (1 + z | grp2) + (1 | grp3)')
    design_metadata = designmatrices_metadata(
        formula,
        make_metadata_lookup([]))
    prior_edits = [
        PriorEdit(['b'], 'b'),
        PriorEdit(['sd'], 'a'),
        PriorEdit(['sd', 'grp2'], 'c'),
        PriorEdit(['sd', 'grp2', 'z'], 'd'),
        PriorEdit(['cor'], 'e'),
        PriorEdit(['cor', 'grp3'], 'f'),
        PriorEdit(['resp', 'sigma'], 'g'),
    ]

    tree = build_prior_tree(formula, design_metadata, getfamily('Normal'), prior_edits)
    pp([('/'.join(path), prior) for path, prior in leaves(tree)])

    # [('b/intercept', 'b'),
    #  ('b/x1', 'b'),
    #  ('b/x2', 'b'),
    #  ('sd/grp1/intercept', 'a'),
    #  ('sd/grp2/intercept', 'c'),
    #  ('sd/grp2/z', 'd'),
    #  ('sd/grp3/intercept', 'a'),
    #  ('cor/grp2', 'e'),
    #  ('cor/grp3', 'f'),
    #  ('resp/sigma', 'g')]

    priors = get_priors(formula, design_metadata, getfamily('Normal'), prior_edits, chk=False)
    pp(priors)
    # {'b': [('b', 3)],
    #  'cor': {'grp2': 'e', 'grp3': 'f'},
    #  'resp': {'sigma': 'g'},
    #  'sd': {'grp1': [('a', 1)], 'grp2': [('c', 1), ('d', 1)], 'grp3': [('a', 1)]}}

    print(check_prior_edit(tree, PriorEdit(['cor', 'grp2'], Prior(getfamily('Normal'), []))))
    # Only LKJ ...



    # BUG:

    tree = Node('parent',
                None,
                [],
                [Node('child', None, [chk_pos_support], [])])
    edit = PriorEdit([], Prior(getfamily('Normal'), [0., 1.]))
    tree = customize_prior(tree, [edit])
    tree = fill(tree)
    # print(tree)

    # check_prior_edit says this edit is OK:
    assert check_prior_edit(tree, edit) is None

    # However, if we manually check the prior on the child node
    # against the check for that node (as returned by `getchecks`),
    # then we find there's an error.
    child_check = getchecks(tree, ['child'])[0]
    child_prior = select(tree, ['child']).prior
    #print(child_check(child_prior))
    assert child_check(child_prior) is not None

    # The problem is that by only collecting checks from the root to
    # the node at which we're setting the prior, we're not ensure that
    # nodes lower down in the tree that inherit the new prior, don't
    # have more local checks that prohibit the prior. Here, the +ve
    # check on child ought to prevent setting a `Normal` at the root,
    # since it would be inherited by the child, leading to a prior
    # that violates the checks.

    # I guess the correct thing to do is to explore the whole sub-tree
    # starting from the node at which the prior is set. This would
    # start with the checks collected while walking from the root to
    # that node, and accumulate checks as the exploration progresses.
    # The prior at each node would be checked along the way. (It
    # probably makes sense to do this once fill has happened.)

    # It may be that this situation won't arise with the default prior
    # tree currently in use, but it makes sense to fix this to avoid
    # running in to hard to debug problems in the future.

if __name__ == '__main__':
    main()
