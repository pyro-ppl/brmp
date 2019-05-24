from collections import namedtuple
from enum import Enum

# This is intended to be independent of Pyro, with a view to
# supporting multiple code gen back-ends eventually.

# TODO: Ensure that families always have a support specified.
Family = namedtuple('Family', 'name params support response')

Param = namedtuple('Param', 'name type')
Type = Enum('Type', 'real pos_real boolean unit_interval corr_cholesky')

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

FAMILIES = [
    Family('Normal',
           [Param('mu', Type.real), Param('sigma', Type.pos_real)],
           Type.real,
           Response('mu', LinkFn.identity)),
    Family('Bernoulli',
           [Param('probs', Type.unit_interval)],
           Type.boolean,
           Response('probs', LinkFn.logit)),
    Family('Cauchy',
           [Param('loc', Type.real), Param('scale', Type.pos_real)],
           Type.real, None),
    Family('HalfCauchy',
           [Param('scale', Type.pos_real)],
           Type.pos_real, None),
    Family('LKJ',
           [Param('eta', Type.pos_real)],
           Type.corr_cholesky, None),
]

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
    return [param for param in family.params if not param.name == family.response.param]

def main():
    pass

if __name__ == '__main__':
    main()
