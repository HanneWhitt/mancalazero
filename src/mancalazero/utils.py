import numpy as np


def fill(length, indexes, values=1):
    mask = np.zeros(length)
    mask[indexes] = values
    return mask


def np_nans(shape, dtype=float):
    a = np.empty(shape)
    a.fill(np.nan)
    a = a.astype(dtype)
    return a


def wrap_assign(a, v, start, end):
    if end > start:
        a[start:end] = v
    else:
        split = a.shape[0] - start
        a[start:] = v[:split]
        a[0:end] = v[split:]