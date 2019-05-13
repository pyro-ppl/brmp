import re
from enum import Enum
from collections import namedtuple

# Tokens
Paren = Enum('Paren', 'L R')
Assoc = Enum('Assoc', 'L R')
Op = namedtuple('Op', 'name assoc precedence')
Var = namedtuple('Var', 'name')

OPS = {
    '+':  Op('+',  Assoc.L, 4),
    '||': Op('||', Assoc.L, 3),
    '|':  Op('|',  Assoc.L, 3),
    '~':  Op('~',  Assoc.L, 2),
}

# AST
Leaf = namedtuple('Leaf', 'value')
Node = namedtuple('Node', 'op l r')

# TODO: Make into classes. Add validation. Add repr.

# TODO: We need to remove duplicate terms and add intercepts by
# default. That probably ought to happen somewhere around here.

Formula = namedtuple('Formula',
                     ['response',   # response column name
                      'pterms',     # list of population level columns
                      'groups'])    # list of groups

Group = namedtuple('Group',
                   ['gterms',       # list of group level columns
                    'column',       # name of grouping column
                    'corr'])        # model correlation between coeffs?

Intercept = namedtuple('Intercept', [])
_1 = Intercept()

def tokenize(inp):
    return [str2token(s) for s in re.findall(r'\b\w+\b|[()~+]|\|\|?', inp)]

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

# Evaluate rpn (as produced by `parse`) to an ast.
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

# Evaluate the rhs of a formula, by mapping a bunch of terms joined
# with `+` to a list. Returns a pair of lists -- one of
# population-level terms and the other of `Group`s.

# The formula y ~ x + x ought to be equal to y ~ x. Interpreting `+`
# here as set union (rather than list concat) is probably a good way
# to implement that.

def evalsum(ast, allow_groups=True):
    if type(ast) == Leaf:
        return [ast.value], []
    elif type(ast) == Node and ast.op == '+':
        tl, gl = evalsum(ast.l, allow_groups)
        tr, gr = evalsum(ast.r, allow_groups)
        return tl + tr, gl + gr
    elif type(ast) == Node and ast.op in ['|', '||'] and allow_groups:
        # This fails when the rhs is not a single variable/term.
        assert type(ast.r) == Leaf
        group_factor = ast.r.value
        # Nesting of groups is not allowed.
        terms, groups = evalsum(ast.l, allow_groups=False)
        assert len(groups) == 0
        return [], [Group(terms, group_factor, ast.op == '|')]
    else:
        # This if/else is not exhaustive, this can occur in regular
        # use. e.g. When nested groups are present.
        raise Exception('unhandled ast')

# Evaluate a formula of the form `y ~ <rhs>`, where the rhs is a sum
# of population terms and groups.
def evalf(ast):
    assert type(ast) == Node and ast.op == '~'
    # The lhs is expected to be a (response) variable
    assert type(ast.l) == Leaf
    pterms, groups = evalsum(ast.r)
    return Formula(ast.l.value, pterms, groups)

def handle_intercept(f):
    def handle(terms):
        return [_1 if t == '1' else t for t in terms]
    return Formula(f.response,
                   handle(f.pterms),
                   [Group(handle(g.gterms), g.column, g.corr) for g in f.groups])

def parse(s):
    return handle_intercept(evalf(rpn2ast(shunt(tokenize(s)))))

def main():
    print(parse('y ~ x1 + x2 + (1 + x3 | x4)'))

if __name__ == '__main__':
    main()
