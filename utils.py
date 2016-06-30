﻿'''
Useful functions for data transfer between CPU and GPU
'''

from collections import OrderedDict

import numpy
import theano
from theano import config


def to_t_float(data):
    '''
    Convert given data to float format used by theano
    '''
    return numpy.asarray(data, dtype=config.floatX)


def init_t_params(params, t_params):
    '''
    Initialize theano parameters from given values
    '''
    for key, value in params.items():
        t_params[key] = theano.shared(value, name=key)


def params_zip(params, t_params):
    '''
    Zip parameters to GPU
    '''
    for key, value in params.items():
        t_params[key].set_value(value)


def params_unzip(t_params):
    '''
    Unzip parameters from GPU
    '''
    params = OrderedDict()
    for key, value in t_params.items():
        params[key] = value.get_value()

    return params