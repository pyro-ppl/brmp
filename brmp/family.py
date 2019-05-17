from collections import namedtuple
from enum import Enum

# This is intended to be independent of Pyro, with a view to
# supporting multiple code gen back-ends eventually.

# This will likely need to record whether the support is continuous or
# discrete, at least.
Family = namedtuple('Family', 'name params response')

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
    Family('Normal', ['mu', 'sigma'], Response('mu', LinkFn.identity)),
    Family('Bernoulli', ['probs'], Response('probs', LinkFn.logit)),
    Family('Cauchy', ['loc', 'scale'], None),
    Family('HalfCauchy', ['scale'], None),
    Family('LKJ', ['eta'], None),
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
    return list(set(family.params) - {family.response.param})

def main():
    pass

if __name__ == '__main__':
    main()
