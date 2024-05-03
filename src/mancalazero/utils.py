import numpy as np


def fill(length, indexes, values=1):
    mask = np.zeros(length)
    mask[indexes] = values
    return mask


def np_nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a