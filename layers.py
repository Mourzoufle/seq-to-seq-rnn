﻿'''
Neural network layers
'''

from collections import OrderedDict

import numpy
import theano
from theano import tensor
from theano import config

from utils import to_t_float, init_t_params


def _concat(prefix, name):
    '''
    Concatenate the name of a parameter with the prefix
    '''
    return '%s_%s' % (prefix, name)


def _slice(tensor_in, idx, n_dim):
    '''
    Fetch part of a tensor of given size in the last dimension
    '''
    if tensor_in.ndim == 3:
        return tensor_in[:, :, idx * n_dim: (idx + 1) * n_dim]
    return tensor_in[:, idx * n_dim: (idx + 1) * n_dim]


def _init_weight(n_dim_in, n_dim_out, scale=0.01):
    '''
    Initialize the weights. Use random samples in normal distribution
    '''
    return scale * numpy.random.randn(n_dim_in, n_dim_out).astype(config.floatX)


def dropout(state_before, use_dropout, trng):
    '''
    Dropout layer
    '''
    return tensor.switch(use_dropout, (state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)), state_before * 0.5)


def dense(state_in, t_params, n_dim_in, n_dim_out, prefix, init):
    '''
    Full-connected layer
    '''
    if init:
        params = OrderedDict()
        params[_concat(prefix, 'W')] = _init_weight(n_dim_in, n_dim_out)
        params[_concat(prefix, 'b')] = numpy.zeros((n_dim_out,)).astype(config.floatX)
        init_t_params(params, t_params)

    return tensor.dot(state_in, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')]


def softmax(state_in):
    '''
    Softmax layer
    '''
    state_out = tensor.exp(state_in - state_in.max(axis=-1, keepdims=True))

    return state_out / state_out.sum(axis=-1, keepdims=True)


def gru(mask, state_below, t_params, n_dim_in, n_dim_out, prefix, init):
    '''
    Gated Recurrent Unit (GRU) layer
    '''
    def _step(_m, _x, _h):
        pre_act = tensor.dot(_h, t_params[_concat(prefix, 'U')]) + _x

        gate_r = tensor.nnet.sigmoid(_slice(pre_act, 0, n_dim_out))
        gate_u = tensor.nnet.sigmoid(_slice(pre_act, 1, n_dim_out))

        _h_new = tensor.tanh(tensor.dot(gate_r * _h, _slice(t_params[_concat(prefix, 'U')], 2, n_dim_out)) + _slice(_x, 2, n_dim_out))
        _h_new = (1. - gate_u) * _h + gate_u * _h_new
        _h_new = _m[:, None] * _h_new + (1. - _m)[:, None] * _h

        return _h_new

    if init:
        params = OrderedDict()
        params[_concat(prefix, 'W')] = numpy.concatenate([_init_weight(n_dim_in, n_dim_out), _init_weight(n_dim_in, n_dim_out), _init_weight(n_dim_in, n_dim_out)], axis=1)
        params[_concat(prefix, 'U')] = numpy.concatenate([_init_weight(n_dim_out, n_dim_out), _init_weight(n_dim_out, n_dim_out), _init_weight(n_dim_out, n_dim_out)], axis=1)
        params[_concat(prefix, 'b')] = numpy.zeros((3 * n_dim_out,)).astype(config.floatX)
        init_t_params(params, t_params)

    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1
    state_below = (tensor.dot(state_below, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')])
    rval, _ = theano.scan(_step, sequences=[mask, state_below], outputs_info=[tensor.alloc(to_t_float(0.), n_samples, n_dim_out)], name=_concat(prefix, '_layer'), n_steps=n_steps)

    return rval


def lstm(mask, state_below, t_params, n_dim_in, n_dim_out, prefix, init):
    '''
    Long Short-Term Memory (LSTM) layer
    '''
    def _step(_m, _x, _h, _c):
        pre_act = tensor.dot(_h, t_params[_concat(prefix, 'U')]) + _x

        gate_i = tensor.nnet.sigmoid(_slice(pre_act, 0, n_dim_out))
        gate_f = tensor.nnet.sigmoid(_slice(pre_act, 1, n_dim_out))
        gate_o = tensor.nnet.sigmoid(_slice(pre_act, 2, n_dim_out))

        _c_new = gate_f * _c + gate_i * tensor.tanh(_slice(pre_act, 3, n_dim_out))
        _c_new = _m[:, None] * _c_new + (1. - _m)[:, None] * _c
        _h_new = gate_o * tensor.tanh(_c_new)
        _h_new = _m[:, None] * _h_new + (1. - _m)[:, None] * _h

        return _h_new, _c_new

    if init:
        params = OrderedDict()
        params[_concat(prefix, 'W')] = numpy.concatenate([_init_weight(n_dim_in, n_dim_out), _init_weight(n_dim_in, n_dim_out), _init_weight(n_dim_in, n_dim_out), _init_weight(n_dim_in, n_dim_out)], axis=1)
        params[_concat(prefix, 'U')] = numpy.concatenate([_init_weight(n_dim_out, n_dim_out), _init_weight(n_dim_out, n_dim_out), _init_weight(n_dim_out, n_dim_out), _init_weight(n_dim_out, n_dim_out)], axis=1)
        params[_concat(prefix, 'b')] = numpy.zeros((4 * n_dim_out,)).astype(config.floatX)
        init_t_params(params, t_params)

    n_steps = state_below.shape[0]
    n_samples = state_below.shape[1] if state_below.ndim == 3 else 1
    state_below = (tensor.dot(state_below, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')])
    rval, _ = theano.scan(_step, sequences=[mask, state_below], outputs_info=[tensor.alloc(to_t_float(0.), n_samples, n_dim_out), tensor.alloc(to_t_float(0.), n_samples, n_dim_out)], name=_concat(prefix, '_layer'), n_steps=n_steps)

    return rval[0]