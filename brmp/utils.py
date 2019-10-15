import traceback

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


class traceback_generated(object):
    """
    Context manager for augmenting exception message when running generated
    code. The particular line from the generated code that resulted in the
    exception is highlighted. This has no effect if the error originated
    from elsewhere in the code.

    :param str code: String representation of the code being run.
    :raises: ModelSpecificationError
    """
    def __init__(self, code):
        self.code = code

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        tb_info = traceback.extract_tb(exc_tb)
        filename, line, fn, _ = tb_info[-1]
        line = line - 1
        # only augment if exception is from generated code.
        if filename == '<string>':
            exc_lines = self.code.split('\n')
            exc_lines = '\n'.join(exc_lines[:line] +
                                  [exc_lines[line] + '\t<<< ==== ERROR ===='] +
                                  exc_lines[line + 1:])
            raise ModelSpecificationError(f'Exception in model code: \n\n {exc_lines}') from exc_type


class ModelSpecificationError(BaseException):
    pass
