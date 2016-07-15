'''
Neural network layers
'''

from collections import OrderedDict

import numpy
import theano
from theano import tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams

from utils import to_floatX, init_t_params


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


def norm_weight(n_dim_in, n_dim_out, scale=0.01):
    '''
    Initialize the weights - Use random samples in normal distribution
    '''
    return scale * numpy.random.randn(n_dim_in, n_dim_out).astype(config.floatX)


def ortho_weight(n_dim_in, n_dim_out, scale=0.01):
    '''
    Initialize the weights - Use a group of orthogonal vectors after SVD factorization
    '''
    mat_u, _, mat_v = numpy.linalg.svd(scale * numpy.random.randn(n_dim_in, n_dim_out).astype(config.floatX), False)
    if n_dim_in >= n_dim_out:
        return mat_u
    else:
        return mat_v


def dropout(state_in, use_dropout, t_rng=None, ratio=0.5):
    '''
    Dropout layer
    '''
    if t_rng is None:
        t_rng = MRG_RandomStreams()

    return tensor.switch(use_dropout, state_in * t_rng.binomial(state_in.shape, p=1 - ratio, dtype=state_in.dtype), state_in * (1 - ratio))


def dense(state_in, t_params, n_dim_in, n_dim_out, prefix):
    '''
    Full-connected layer
    '''
    if _concat(prefix, 'W') not in t_params:
        params = OrderedDict()
        params[_concat(prefix, 'W')] = ortho_weight(n_dim_in, n_dim_out)
        params[_concat(prefix, 'b')] = numpy.zeros((n_dim_out,), config.floatX)
        init_t_params(params, t_params)

    return tensor.dot(state_in, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')]


def embedding(state_in, t_params, n_dim_in, n_dim_out, prefix):
    '''
    Word embedding layer - Return a matrix that nay need to be reshaped
    '''
    if _concat(prefix, 'W') not in t_params:
        params = OrderedDict()
        params[_concat(prefix, 'W')] = ortho_weight(n_dim_in, n_dim_out)
        init_t_params(params, t_params)

    return t_params[_concat(prefix, 'W')][state_in.flatten()]


def gru(mask, state_in, t_params, n_dim_in, n_dim_out, prefix, one_step=False, init_h=None):
    '''
    Gated Recurrent Unit (GRU) layer
    '''
    def _step(_mask, _state_in, _prev_h):
        _pre_act = tensor.dot(_prev_h, t_params[_concat(prefix, 'U')])

        _gate_r = tensor.nnet.sigmoid(_slice(_pre_act, 0, n_dim_out) + _slice(_state_in, 0, n_dim_out))
        _gate_u = tensor.nnet.sigmoid(_slice(_pre_act, 1, n_dim_out) + _slice(_state_in, 1, n_dim_out))

        _next_h = tensor.tanh(_gate_r * _slice(_pre_act, 2, n_dim_out) + _slice(_state_in, 2, n_dim_out))
        _next_h = (1. - _gate_u) * _prev_h + _gate_u * _next_h
        _next_h = _mask[:, None] * _next_h + (1. - _mask)[:, None] * _prev_h

        return _next_h

    if _concat(prefix, 'W') not in t_params:
        params = OrderedDict()
        params[_concat(prefix, 'W')] = numpy.concatenate([ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out)], 1)
        params[_concat(prefix, 'U')] = numpy.concatenate([ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out)], 1)
        params[_concat(prefix, 'b')] = numpy.zeros((3 * n_dim_out,), config.floatX)
        init_t_params(params, t_params)

    state_in = (tensor.dot(state_in, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')])
    if init_h is None:
        init_h = tensor.alloc(to_floatX(0.), state_in.shape[-2], n_dim_out)
    if one_step:
        return _step(mask, state_in, init_h)
    else:
        state_out, _ = theano.scan(_step, [mask, state_in], [init_h])
        return state_out


def lstm(mask, state_in, t_params, n_dim_in, n_dim_out, prefix, one_step=False, init_h=None):
    '''
    Long Short-Term Memory (LSTM) layer
    '''
    def _step(_mask, _state_in, _prev_h, _prev_c):
        _pre_act = tensor.dot(_prev_h, t_params[_concat(prefix, 'U')]) + _state_in

        _gate_i = tensor.nnet.sigmoid(_slice(_pre_act, 0, n_dim_out))
        _gate_f = tensor.nnet.sigmoid(_slice(_pre_act, 1, n_dim_out))
        _gate_o = tensor.nnet.sigmoid(_slice(_pre_act, 2, n_dim_out))

        _next_c = _gate_f * _prev_c + _gate_i * tensor.tanh(_slice(_pre_act, 3, n_dim_out))
        _next_c = _mask[:, None] * _next_c + (1. - _mask)[:, None] * _prev_c
        _next_h = _gate_o * tensor.tanh(_next_c)
        _next_h = _mask[:, None] * _next_h + (1. - _mask)[:, None] * _prev_h

        return _next_h, _next_c

    if _concat(prefix, 'W') not in t_params:
        params = OrderedDict()
        params[_concat(prefix, 'W')] = numpy.concatenate([ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out)], 1)
        params[_concat(prefix, 'U')] = numpy.concatenate([ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out)], 1)
        params[_concat(prefix, 'b')] = numpy.zeros((4 * n_dim_out,), config.floatX)
        init_t_params(params, t_params)

    state_in = (tensor.dot(state_in, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')])
    if init_h is None:
        init_h = tensor.alloc(to_floatX(0.), state_in.shape[-2], n_dim_out)
    if one_step:
        state_out, _ = _step(mask, state_in, init_h, tensor.zeros_like(init_h))
        return state_out
    else:
        [state_out, _], _ = theano.scan(_step, [mask, state_in], [init_h, tensor.zeros_like(init_h)])
        return state_out
