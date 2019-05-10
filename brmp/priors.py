from collections import namedtuple
from pprint import pprint as pp

import pandas as pd

from pyro.contrib.brm.formula import Formula, _1
from pyro.contrib.brm.design import dfmetadata, designmatrix_metadata, DesignMeta, PopulationMeta, GroupMeta

Node = namedtuple('Node', 'name prior children')

def leaf(name):
    return Node(name, None, [])

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

def select(node, path):
    if len(path) == 0:
        return node
    else:
        name = path[0]
        selected_node = next((n for n in node.children if n.name == name), None)
        if selected_node is None:
            raise Exception('Invalid path')
        return select(selected_node, path[1:])

def edit(node, path, f):
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
        return Node(node.name, node.prior, children)

# TODO: Add "class" for correlation matrices.

# TODO: Figure out how to incorporate priors on the response
# distribution.

def default_prior(design_metadata):
    assert type(design_metadata) == DesignMeta
    assert type(design_metadata.population) == PopulationMeta
    assert type(design_metadata.groups) == list
    assert all(type(gm) == GroupMeta for gm in design_metadata.groups)
    ptree = Node('b', None, [leaf(name) for name in design_metadata.population.coefs])
    gtrees = [Node(gm.name, None, [leaf(name) for name in gm.coefs]) for gm in design_metadata.groups]
    # TODO: I need to ensure every coef has a default prior in order
    # to generate code. This is OK for now, but it doesn't match brms.
    tree = Node('root', Prior('Normal', [0., 1.]), [ptree, Node('sd', None, gtrees)])
    return tree

# TODO: This ought to warn/error when an element of `priors` has a
# path that doesn't correspond to a node in the tree.

def customize_prior(tree, prior_edits):
    assert type(tree) == Node
    assert type(prior_edits) == list
    assert all(type(p) == PriorEdit for p in prior_edits)
    for p in prior_edits:
        tree = edit(tree, p.path,
                    lambda n: Node(n.name, p.prior, n.children))
    return tree

def get_prior(design_metadata, prior_edits):
    return fill(customize_prior(default_prior(design_metadata), prior_edits))

# TODO: dedup
def join(lists):
    return sum(lists, [])

def tree2list(node, path=[]):
    return [('/'.join(path), node.prior)] + join(tree2list(n, path+[n.name]) for n in node.children)


# `fill` populates the `prior` property of all nodes in a tree. Each
# node uses its own `prior` value if set, otherwise the first `prior`
# value encountered when walking up the tree from the node is used.
# (This is the behaviour, not the implementation.)
def fill(node, default=None):
    prior = node.prior if not node.prior is None else default
    return Node(node.name, prior, [fill(n, prior) for n in node.children])


def leaves(node, path=[]):
    if len(node.children) == 0:
        # Return `node.prior` rather than the entire `node`, since the
        # other properties are recoverable from path and fact that
        # node is a leaf.
        return [(path, node.prior)]
    else:
        return join(leaves(n, path + [n.name]) for n in node.children)

# e.g.
# contig(list('abb')) == [('a', 0, 1), ('b', 1, 3)]
def contig(xs, chk=True):
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
    segments = [(x, ix[0], ix[0] + len(ix)) for (x, ix) in segments]
    # Some (partial) correctness checks.
    for val, start, end in segments:
        assert all(x == val for x in xs[start:end])
    if chk:
        segment_vals = [val for (val, _, _) in segments]
        assert norepeats(segment_vals)
    return segments

def norepeats(xs):
    return len(contig(xs, False)) == len(xs)


# TODO: Rename.
# This is an example of using this stuff to compute a description of
# vectorized priors as might be used for codegen.
def foobar(tree, path):
    return contig([n.prior for n in select(tree, path).children])


def get_priors(design_metadata, priors):
    assert type(design_metadata) == DesignMeta
    # TODO: maybe get_prior should be renamed to `build_prior_tree`?
    # (Since get_prior is similar to `get_priors` though they are a
    # bit different. Also, `get_priors` seems like an ok name for this
    # when used from codegen.)
    tree = get_prior(design_metadata, priors)
    # TODO: Add sd priors; incorporate into codegen.
    return dict(b=foobar(tree, ['b']))

def main():
    p = get_prior(
        DesignMeta(
            PopulationMeta(['intercept', 'x1', 'x2']),
            [
                GroupMeta('grp1', ['intercept']),
                GroupMeta('grp2', ['intercept', 'z']),
                GroupMeta('grp3', ['intercept'])
            ]),
        # priors
        [
            PriorEdit(['b'], 'b'),
            PriorEdit(['sd'], 'a'),
            PriorEdit(['sd', 'grp2'], 'c'),
            PriorEdit(['sd', 'grp2', 'z'], 'd'),
        ])

    pp([('/'.join(path), prior) for path, prior in leaves(p)])

    # [('b/intercept',       'b'),
    #  ('b/x1',              'b'),
    #  ('b/x2',              'b'),
    #  ('sd/grp1/intercept', 'a'),
    #  ('sd/grp2/intercept', 'c'),
    #  ('sd/grp2/z',         'd'),
    #  ('sd/grp3/intercept', 'a')]

    print(foobar(p, ['b']))
    # All coefs use prior 'b':
    # [('b', 0, 3)]
    print(foobar(p, ['sd', 'grp2']))
    # First coef uses prior 'c', the second 'd'.
    # [('c', 0, 1), ('d', 1, 2)]

if __name__ == '__main__':
    main()
