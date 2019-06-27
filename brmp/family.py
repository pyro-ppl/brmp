from collections import namedtuple
from enum import Enum
import inspect
from functools import partial

# This is intended to be independent of Pyro, with a view to
# supporting multiple code gen back-ends eventually.

# TODO: Ensure that families always have a support specified.
Family = namedtuple('Family', 'name params support response')

# TODO: Check that `value` is in `type` (or None).
Param = namedtuple('Param', 'name type value')
def param(name, typ):
    return Param(name, typ, None)

def mktype(*args):
    out = namedtuple(*args)
    def eq(a, b):
        return type(a) == type(b) and tuple(a) == tuple(b)
    out.__eq__ = eq
    out.__ne__ = lambda a, b: not eq(a, b)
    return out

Type = dict(
    Real         = mktype('Real', ''),
    PosReal      = mktype('PosReal', ''),
    Boolean      = mktype('Boolean', ''),
    UnitInterval = mktype('UnitInterval', ''),
    CorrCholesky = mktype('CorrCholesky', ''),
    IntegerRange = mktype('IntegerRange', 'lb ub'),
)

def istype(ty):
    return type(ty) in Type.values()


# Inverse might also be called recip(rocal).
LinkFn = Enum('LinkFn', 'identity logit inverse')

Response = namedtuple('Response', 'param linkfn')

# TODO: Code generation currently assumes that the parameters here
# appear in the order expected by Pyro distribution constructors. This
# is bad, because I'm aiming for a backend agnostic implementation of
# families. (Because the order it used, it does mean the names doesn't
# have match, so I can use names here that ensure that the args. you
# need to give when customising priors is similar to brms.)

# TODO: Add more response families.

def const(x):
    def f(*args):
        return x
    return f

FAMILIES = [
    Family('Normal',
           [param('mu', Type['Real']()), param('sigma', Type['PosReal']())],
           const(Type['Real']()),
           Response('mu', LinkFn.identity)),
    Family('Bernoulli',
           [param('probs', Type['UnitInterval']())],
           const(Type['Boolean']()),
           Response('probs', LinkFn.logit)),
    Family('Cauchy',
           [param('loc', Type['Real']()), param('scale', Type['PosReal']())],
           const(Type['Real']()), None),
    Family('HalfCauchy',
           [param('scale', Type['PosReal']())],
           const(Type['PosReal']()), None),
    Family('LKJ',
           [param('eta', Type['PosReal']())],
           const(Type['CorrCholesky']()), None),
    Family('Binomial',
           [param('num_trials', Type['IntegerRange'](0, None)),
            param('probs', Type['UnitInterval']())],
           lambda num_trials: Type['IntegerRange'](0, num_trials),
           Response('probs', LinkFn.logit)),
]

def apply1(family, name, value):
    if name not in [p.name for p in family.params]:
        raise Exception('Unknown parameter "{}"'.format(name))
    def setval(param, value):
        return Param(param.name, param.type, value)
    params = [setval(param, value) if param.name == name else param
              for param in family.params]
    if name in inspect.getfullargspec(family.support).args:
        support = partial(family.support, **{name: value})
    else:
        support = family.support
    return Family(family.name, params, support, family.response)

# This could be __call__ on a Family class.
def apply(family, **kwargs):
    for name, value in kwargs.items():
        family = apply1(family, name, value)
    return family

def support_depends_on_args(family):
    return len(inspect.getfullargspec(family.support).args) > 0

def fully_applied(family):
    return all(param.value is not None for param in family.params)

def args(family):
    return [param.value for param in family.params]

def family_repr(family):
    params = ', '.join('{}={}'.format(param.name, param.value)
                       for param in family.params
                       if not param.value is None)
    return '{}({})'.format(family.name, params)

def lookup(items, name):
    for item in items:
        if item.name == name:
            return item
    raise Exception('Family "{}" not found.'.format(name))

def getfamily(name):
    return lookup(FAMILIES, name)

def nonlocparams(family):
    assert type(family) == Family
    assert family.response is not None
    return [param for param in family.params
            if not (param.name == family.response.param or param.value is not None)]

# Note that the location parameter is always called "mu" in the list
# returned by this function.
def free_param_names(family):
    assert type(family) == Family
    assert family.response is not None
    return ['mu' if param.name == family.response.param else param.name
            for param in family.params if param.value is None]

def main():
    pass

if __name__ == '__main__':
    main()
