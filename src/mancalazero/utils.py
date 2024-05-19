import numpy as np
import torch


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


def custom_collate(batch):
    batch = [torch.from_numpy(np.vstack(x)) for x in zip(*batch)]
    batch[1] = batch[1].view(-1)
    return batch


def add_dirichlet_noise(p, alpha, noise_fraction):
    noise = np.random.dirichlet([alpha]*len(p))
    with_noise = (1 - noise_fraction)*p + noise_fraction*noise
    return with_noise