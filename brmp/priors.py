from collections import namedtuple
from pprint import pprint as pp

import pandas as pd
import torch.distributions.constraints as constraints
import pyro.distributions as dists

from pyro.contrib.brm.utils import join
from pyro.contrib.brm.formula import Formula, parse
from pyro.contrib.brm.design import designmatrices_metadata, DesignMeta, PopulationMeta, GroupMeta, make_metadata_lookup

Node = namedtuple('Node', 'name prior checks children')

def leaf(name):
    return Node(name, None, [], [])

# e.g. Prior('Normal', [0., 1.]).
# Also see the related `gendist` function in codegen.py.

# TODO: This currently requires `parameters` to be a list of floats.
# This ought to be checked.
Prior = namedtuple('Prior', 'family parameters')

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

# TODO: Figure out how to incorporate priors on the response
# distribution.

# TODO: Match default priors used by brms. (An improper uniform is
# used for `b`. A Half Student-t here is used for priors on standard
# deviations, with its scale derived from the data.

def default_prior(formula, design_metadata):
    assert type(formula) == Formula
    assert type(design_metadata) == DesignMeta
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
    return Node('root', None, [chk_known_dist], [
        Node('b',   Prior('Cauchy', [0., 1.]), [], b_children),
        Node('sd',  Prior('HalfCauchy', [3.]), [chk_pos_support], sd_children),
        Node('cor', Prior('LKJ', [1.]),        [chk_lkj], cor_children)])

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
def build_prior_tree(formula, design_metadata, prior_edits):
    return fill(customize_prior(default_prior(formula, design_metadata), prior_edits))


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

def get_priors(formula, design_metadata, prior_edits, chk=True):
    assert type(formula) == Formula
    assert type(design_metadata) == DesignMeta
    assert type(prior_edits) == list
    tree = build_prior_tree(formula, design_metadata, prior_edits)
    if chk:
        check_prior_edits(tree, prior_edits)
    def get(path):
        return contig([n.prior for n in select(tree, path).children])
    return dict(
        b=get(['b']),
        sd=dict((group_meta.name, get(['sd', group_meta.name]))
                for group_meta in design_metadata.groups),
        cor=dict((n.name, n.prior) for n in select(tree, ['cor']).children))

# Sanity checks

def chk(error):
    def decorate(predicate):
        def f(prior):
            retval = predicate(prior)
            assert type(retval) == bool
            return None if retval else error
        f.predicate = predicate
        return f
    return decorate

@chk('A distribution with support on only the positive reals expected here.')
def chk_pos_support(prior):
    dist = dists.__getattribute__(prior.family)
    return dist.support == constraints.positive

@chk('Unknown distribution family.')
def chk_known_dist(prior):
    if chk_lkj.predicate(prior):
        return True
    try:
        dists.__getattribute__(prior.family)
        return True
    except AttributeError:
        return False

@chk('Only the LKJ(...) family is supported here.')
def chk_lkj(prior):
    return prior.family == 'LKJ'

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
    ]

    tree = build_prior_tree(formula, design_metadata, prior_edits)
    pp([('/'.join(path), prior) for path, prior in leaves(tree)])

    # [('b/intercept',       'b'),
    #  ('b/x1',              'b'),
    #  ('b/x2',              'b'),
    #  ('sd/grp1/intercept', 'a'),
    #  ('sd/grp2/intercept', 'c'),
    #  ('sd/grp2/z',         'd'),
    #  ('sd/grp3/intercept', 'a'),
    #  ('cor/grp2',          'e'),
    #  ('cor/grp3',          'f')]

    priors = get_priors(formula, design_metadata, prior_edits, chk=False)
    pp(priors)
    # {'b': [('b', 3)],
    #  'cor': {'grp2': 'e', 'grp3': 'f'},
    #  'sd': {'grp1': [('a', 1)], 'grp2': [('c', 1), ('d', 1)], 'grp3': [('a', 1)]}}

    print(check_prior_edit(tree, PriorEdit([], Prior('Normal2', []))))
    # Unknown distribution ...

    print(check_prior_edit(tree, PriorEdit(['cor', 'grp2'], Prior('Normal', []))))
    # Only LKJ ...



if __name__ == '__main__':
    main()
