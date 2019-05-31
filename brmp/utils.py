def join(lists):
    return sum(lists, [])

def unzip(pairs):
    if len(pairs) == 0:
        return [], []
    else:
        return zip(*pairs)
