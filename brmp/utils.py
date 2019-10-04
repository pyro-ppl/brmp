def join(lists):
    return sum(lists, [])

def unzip(pairs):
    if len(pairs) == 0:
        return [], []
    else:
        return zip(*pairs)

# Helpers for working with arrays of samples. These work with both
# numpy and torch arrays.

# (num_chains, num_samples, ...) -> (num_chains * num_samples, ...)
def flatten(arr):
    return arr.reshape((arr.shape[0] * arr.shape[1],) + arr.shape[2:])

# (num_chains * num_samples, ...) -> (num_chains, num_samples, ...)
def unflatten(arr, num_chains, num_samples):
    return arr.reshape((num_chains, num_samples) + arr.shape[1:])
