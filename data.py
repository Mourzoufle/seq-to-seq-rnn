'''
Operations on the dataset
'''

import copy
import json

import numpy

from utils import to_floatX


def load_data(path_data, path_mat, n_dim_y, max_samples):
    '''
    Load dataset
    '''
    with open(path_data, 'r') as file_in:
        samples = json.load(file_in)
        for sample in samples:
            for i in range(1, len(sample[1])):
                sample[1][0].extend(sample[1][i])
            sample[1] = sample[1][0]
            n_dim_y = max(n_dim_y, max(sample[1]) + 1)

        if max_samples > 0:
            samples = samples[: max_samples]

    return samples, numpy.load(path_mat), n_dim_y


def load_batch(samples, batch, mat):
    '''
    Fetch and expand data of a batch in NumPy ndarray format
    '''
    x = []
    y = []
    for idx in batch:
        x.append(copy.copy(samples[idx][0]))
        y.append(copy.copy(samples[idx][1]))
    max_len_x = max(len(row) for row in x)
    max_len_y = max(len(row) for row in y)

    mask_x = []
    mask_y = []
    n_samples = len(batch)
    for i in range(n_samples):
        mask_x.append([1] * len(x[i]))
        x[i] += [0] * (max_len_x - len(x[i]))
        mask_x[i] += [0] * (max_len_x - len(mask_x[i]))
        mask_y.append([1] * len(y[i]))
        y[i] += [0] * (max_len_y - len(y[i]))
        mask_y[i] += [0] * (max_len_y - len(mask_y[i]))
    x = mat[numpy.asarray(x).flatten()].reshape(n_samples, max_len_x, mat.shape[-1])

    return numpy.swapaxes(to_floatX(x), 0, 1), numpy.asarray(mask_x, 'int8').T, numpy.asarray(y, 'int32').T, numpy.asarray(mask_y, 'int8').T


def get_batches(n_samples, batch_size, shuffle=False):
    '''
    Get batches of data
    '''
    indices = numpy.arange(n_samples, dtype='int32')
    if shuffle:
        numpy.random.shuffle(indices)
    batches = []
    idx_start = 0
    while idx_start < n_samples:
        batches.append(indices[idx_start: idx_start + batch_size])
        idx_start += batch_size

    return batches
