from collections import namedtuple

# TODO: Add a parser.
# TODO: Make into classes. Add validation. Add repr.
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
