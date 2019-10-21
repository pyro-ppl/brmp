import inspect
from collections import namedtuple
from enum import Enum
from functools import partial

# This is intended to be independent of Pyro, with a view to
# supporting multiple code gen back-ends eventually.

# TODO: Ensure that families always have a support specified.
Family = namedtuple('Family', 'name params support link')
Family.__call__ = lambda self, *args, **kwargs: apply(self, *args, **kwargs)
Family.__repr__ = lambda self: family_repr(self)

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
    Real=mktype('Real', ''),
    PosReal=mktype('PosReal', ''),
    Boolean=mktype('Boolean', ''),
    UnitInterval=mktype('UnitInterval', ''),
    CorrCholesky=mktype('CorrCholesky', ''),
    IntegerRange=mktype('IntegerRange', 'lb ub'),
)


def istype(ty):
    return type(ty) in Type.values()


# Inverse might also be called recip(rocal).
LinkFn = Enum('LinkFn', 'identity logit inverse')

Link = namedtuple('Link', 'param fn')


# TODO: Code generation currently assumes that the parameters appear
# here in the order expected by Pyro + NumPyro distribution
# constructors. This is bad, because I'm aiming for a backend agnostic
# implementation of families. Because the order *is* used, it means
# the names here don't have match those in (Num)Pyro, so I can use
# names that match brms. (e.g. The sd parameter of a Normal response
# is called sigma when specifiying priors/inspecting marginals, etc.)

# TODO: Add more response families.

def const(x):
    def f(*args):
        return x

    return f


Normal = Family('Normal',
                [param('mu', Type['Real']()), param('sigma', Type['PosReal']())],
                const(Type['Real']()),
                Link('mu', LinkFn.identity))
"""
:param mu: mean
:param sigma: standard deviation
"""
Bernoulli = Family('Bernoulli',
                   [param('probs', Type['UnitInterval']())],
                   const(Type['Boolean']()),
                   Link('probs', LinkFn.logit))
"""
:param probs: success probability
"""
Cauchy = Family('Cauchy',
                [param('loc', Type['Real']()), param('scale', Type['PosReal']())],
                const(Type['Real']()), None)
"""
:param loc: location
:param scale: scale
"""
HalfCauchy = Family('HalfCauchy',
                    [param('scale', Type['PosReal']())],
                    const(Type['PosReal']()), None)
"""
:param scale: scale
"""
LKJ = Family('LKJ',
             [param('eta', Type['PosReal']())],
             const(Type['CorrCholesky']()), None)
"""
:param eta: shape
"""
Binomial = Family('Binomial',
                  [param('num_trials', Type['IntegerRange'](0, None)),
                   param('probs', Type['UnitInterval']())],
                  lambda num_trials: Type['IntegerRange'](0, num_trials),
                  Link('probs', LinkFn.logit))
"""
:param num_trials: number of trials
:param probs: success probability
"""
HalfNormal = Family('HalfNormal',
                    [param('sigma', Type['PosReal']())],
                    const(Type['PosReal']()),
                    None)
"""
:param sigma: scale
"""


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
    return Family(family.name, params, support, family.link)


# This could be __call__ on a Family class.
def apply(family, *args, **kwargs):
    free_param_names = [param.name
                        for param in family.params
                        if param.value is None]
    for name, value in zip(free_param_names, args):
        family = apply1(family, name, value)
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
                       if param.value is not None)
    return '{}({})'.format(family.name, params)


def nonlocparams(family):
    assert type(family) == Family
    assert family.link is not None
    return [param for param in family.params
            if not (param.name == family.link.param or param.value is not None)]


# Note that the location parameter is always called "mu" in the list
# returned by this function.
def free_param_names(family):
    assert type(family) == Family
    assert family.link is not None
    return ['mu' if param.name == family.link.param else param.name
            for param in family.params if param.value is None]
