import re
from enum import Enum
from collections import namedtuple
import itertools

from brmp.utils import join

# Maintains order.
def unique(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

class OrderedSet():
    def __init__(self, *items, items_are_unique=False):
        # For most methods on this class `items` could be an arbitrary
        # iterable. However, for `union`, we need any two ordered sets
        # to have `items` be of the same type, in order to
        # straight-forwardly concatenate them with `+`. I'll use a
        # tuple, since it also has the benefit of been immutable, but
        # list would work too.
        self.items = tuple(items if items_are_unique else unique(items))
        self.fset = frozenset(self.items)
        assert len(self.fset) == len(self.items) # `unique` ensures nodups
    def __hash__(self):
        return hash((OrderedSet, self.fset))
    def __eq__(self, other):
        return self.fset == other.fset
    def __iter__(self):
        return self.items.__iter__()
    def __next__(self):
        return self.items.__next__()
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]
    def __repr__(self):
        return '<{}>'.format(','.join(str(item) for item in self.items))
    def union(self, other):
        items = self.items + tuple(x for x in other.items if not x in self.fset)
        return OrderedSet(*items, items_are_unique=True)


# Tokens
Paren = Enum('Paren', 'L R')
Assoc = Enum('Assoc', 'L R')
Op = namedtuple('Op', 'name assoc precedence')
Var = namedtuple('Var', 'name')

OPS = {
    ':':  Op(':',  Assoc.L, 5),
    '+':  Op('+',  Assoc.L, 4),
    '||': Op('||', Assoc.L, 3),
    '|':  Op('|',  Assoc.L, 3),
    '~':  Op('~',  Assoc.L, 2),
}

# AST
Leaf = namedtuple('Leaf', 'value')
Node = namedtuple('Node', 'op l r')

# TODO: Make into classes. Add validation. Add repr.

# TODO: Add intercepts by default. That probably ought to happen
# somewhere around here.

# TODO: Check for (and disallow) multiple groups using the same
# grouping column. I assume that this is the case elsewhere in the
# package. (e.g. Specifying priors for a particular group is done
# using the name of the grouping column, which would be ambiguous
# without the assumption. brms does this too.

Formula = namedtuple('Formula',
                     ['response',   # response column name
                      'terms',      # an OrderedSet of population level terms
                      'groups'])    # list of groups

Group = namedtuple('Group',
                   ['terms',        # an OrderedSet of group-level terms
                    'columns',      # names of grouping columns
                    'corr'])        # model correlation between coeffs?


# TODO: Make it possible to union terms directly? (Could use in the
# `:` case of eval.)
Term = namedtuple('Term',
                  ['factors']) # Factors in the Patsy sense. An OrderedSet.
_1 = Term(OrderedSet()) # Intercept

def allfactors(formula):
    assert type(formula) == Formula
    def all_from_terms(terms):
        return join(list(term.factors) for term in terms)
    return ([formula.response] +
            all_from_terms(formula.terms) +
            join(all_from_terms(group.terms) + group.columns for group in formula.groups))

def tokenize(inp):
    return [str2token(s) for s in re.findall(r'\b\w+\b|[()~+:]|\|\|?', inp)]

def str2token(s):
    if s in OPS:
        return OPS[s]
    elif s == '(':
        return Paren.L
    elif s == ')':
        return Paren.R
    else:
        return Var(s)

# https://en.wikipedia.org/wiki/Shunting-yard_algorithm
# TODO: Add better error handling. (Wiki article has some extra checks
# I skipped.)
# TODO: Use stack/queue with correct asymptotic performance.
def shunt(tokens):
    opstack = []
    output = []
    for token in tokens:
        if type(token) == Var:
            output.append(token)
        elif type(token) == Op:
            while (len(opstack) > 0 and opstack[-1] != Paren.L and
                   (opstack[-1].precedence > token.precedence or
                    (opstack[-1].precedence == token.precedence and opstack[-1].assoc == Assoc.L))):
                output.append(opstack.pop())
            opstack.append(token)
        elif token == Paren.L:
            opstack.append(token)
        elif token == Paren.R:
            while opstack[-1] != Paren.L:
                output.append(opstack.pop())
            assert opstack[-1] == Paren.L
            opstack.pop()
        else:
            raise Exception('unhandled token type')
    while opstack:
        output.append(opstack.pop())
    return output

# Evaluate rpn (as produced by `shunt`) to an ast.
def rpn2ast(tokens):
    out = []
    for token in tokens:
        if type(token) == Var:
            out.append(Leaf(token.name))
        elif type(token) == Op:
            right = out.pop()
            left = out.pop()
            out.append(Node(token.name, left, right))
        else:
            # No parens once in rpn.
            raise Exception('unhandled token type')
    assert len(out) == 1
    return out[0]

# Returns an ordered set of population-level terms and a list of
# groups.
def eval_rhs(ast, allow_groups=True):
    if type(ast) == Leaf:
        if ast.value == "1":
            return OrderedSet(_1), []
        else:
            return OrderedSet(Term(OrderedSet(ast.value))), []
    elif type(ast) == Node and ast.op == '+':
        termsl, groupsl = eval_rhs(ast.l, allow_groups)
        termsr, groupsr = eval_rhs(ast.r, allow_groups)
        return termsl.union(termsr), groupsl + groupsr
    elif type(ast) == Node and ast.op == ':':
        # lme4/brms say a formula has the general form:
        # response ~ pterms + (gterms | group) + ...
        # This suggests the interaction between groups is not
        # possible. (Which is good, because I don't know what the
        # semantics would be.) However, neither packages complains if
        # you write something like `y ~ (a | b) : (b | a)`, which is
        # odd. Here we don't allow interactions between groups.
        termsl, groupsl = eval_rhs(ast.l, allow_groups=False)
        termsr, groupsr = eval_rhs(ast.r, allow_groups=False)
        assert len(groupsl) == 0
        assert len(groupsr) == 0
        terms = [Term(tl.factors.union(tr.factors))
                 for tl, tr in itertools.product(termsl, termsr)]
        return OrderedSet(*terms), []
    elif type(ast) == Node and ast.op in ['|', '||'] and allow_groups:
        group_factors = eval_group_rhs(ast.r)
        # Nesting of groups is not allowed.
        terms, groups = eval_rhs(ast.l, allow_groups=False)
        assert len(groups) == 0
        return OrderedSet(), [Group(terms, group_factors, ast.op == '|')]
    else:
        # This if/else is not exhaustive, this can occur in regular
        # use. e.g. When nested groups are present.
        raise Exception('unhandled ast')

# Evaluate the expression to the right of the `|` or `||` in a group
# to a list of factor names.
# e.g. `a:b:c` -> ['a', 'b', 'c']

# TODO: Have `a:a` evaluate to `a`. (By using OrderedSet?)
def eval_group_rhs(ast):
    if type(ast) == Leaf:
        return [ast.value]
    elif type(ast) == Node and ast.op == ':':
        return eval_group_rhs(ast.l) + eval_group_rhs(ast.r)
    else:
        # TODO: Better error. Catch and re-throw within `eval_rhs` so
        # that the whole group can be included in the message?
        raise Exception('unhandled ast')

# Evaluate a formula of the form `y ~ <rhs>`, where the rhs is a sum
# of population terms and groups.
def evalf(ast):
    assert type(ast) == Node and ast.op == '~'
    # The lhs is expected to be a (response) variable
    assert type(ast.l) == Leaf
    terms, groups = eval_rhs(ast.r)
    return Formula(ast.l.value, terms, groups)

def parse(s):
    return evalf(rpn2ast(shunt(tokenize(s))))

def main():
    print(parse('y ~ x1 + x2 + (1 + x3 | x4) + x5:x6'))

if __name__ == '__main__':
    main()
