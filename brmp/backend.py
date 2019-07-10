from collections import namedtuple

# For now, assume that each back end provides a single inference algorithm.
Backend = namedtuple('Backend', 'name gen infer from_numpy to_numpy')

# We could have a class that wraps a (function, code) pair, making the
# code available via a code property and the function available via
# __call__. `Model` could also be callable. Too cute?
Model = namedtuple('Model', [
    'fn', 'code',
    'inv_link_fn', 'inv_link_code',
    'expected_response_fn', 'expected_response_code',
])

def apply_default_hmc_args(iter, warmup):
    iter = 10 if iter is None else iter
    warmup = iter // 2 if warmup is None else warmup
    return iter, warmup
