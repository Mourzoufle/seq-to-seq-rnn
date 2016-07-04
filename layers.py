'''
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


def norm_weight(n_dim_in, n_dim_out, scale=0.01):
    '''
    Initialize the weights - Use random samples in normal distribution
    '''
    return scale * numpy.random.randn(n_dim_in, n_dim_out).astype(config.floatX)


def ortho_weight(n_dim_in, n_dim_out, scale=0.01):
    '''
    Initialize the weights - Use a group of orthogonal vectors after SVD factorization
    '''
    u, _, v = numpy.linalg.svd(scale * numpy.random.randn(n_dim_in, n_dim_out).astype(config.floatX), full_matrices=True)
    if n_dim_in >= n_dim_out:
        return u
    else:
        return v


def dropout(state_in, use_dropout, trng, ratio=0.5):
    '''
    Dropout layer
    '''
    return tensor.switch(use_dropout, (state_in * trng.binomial(state_in.shape, p=ratio, n=1, dtype=state_in.dtype)), state_in * ratio)


def dense(state_in, t_params, n_dim_in, n_dim_out, prefix):
    '''
    Full-connected layer
    '''
    if not t_params.has_key(_concat(prefix, 'b')):
        params = OrderedDict()
        params[_concat(prefix, 'W')] = ortho_weight(n_dim_in, n_dim_out)
        params[_concat(prefix, 'b')] = numpy.zeros((n_dim_out,)).astype(config.floatX)
        init_t_params(params, t_params)

    return tensor.dot(state_in, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')]


def gru(mask, state_in, t_params, n_dim_in, n_dim_out, prefix, one_step=False, state_out=None):
    '''
    Gated Recurrent Unit (GRU) layer
    '''
    def _step(_m, _x, _h):
        pre_act = tensor.dot(_h, t_params[_concat(prefix, 'U')])

        gate_r = tensor.nnet.sigmoid(_slice(pre_act, 0, n_dim_out) + _slice(_x, 0, n_dim_out))
        gate_u = tensor.nnet.sigmoid(_slice(pre_act, 1, n_dim_out) + _slice(_x, 1, n_dim_out))

        _h_new = tensor.tanh(gate_r * _slice(pre_act, 2, n_dim_out) + _slice(_x, 2, n_dim_out))
        _h_new = (1. - gate_u) * _h + gate_u * _h_new
        _h_new = _m[:, None] * _h_new + (1. - _m)[:, None] * _h

        return _h_new

    if not t_params.has_key(_concat(prefix, 'b')):
        params = OrderedDict()
        params[_concat(prefix, 'W')] = numpy.concatenate([ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out)], axis=1)
        params[_concat(prefix, 'U')] = numpy.concatenate([ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out)], axis=1)
        params[_concat(prefix, 'b')] = numpy.zeros((3 * n_dim_out,)).astype(config.floatX)
        init_t_params(params, t_params)

    n_steps = state_in.shape[0]
    n_samples = state_in.shape[1] if state_in.ndim == 3 else 1
    state_in = (tensor.dot(state_in, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')])
    if state_out is None:
        state_out = tensor.alloc(to_t_float(0.), n_samples, n_dim_out)
    if one_step:
        return _step(mask, state_in, state_out)
    else:
        rval, _ = theano.scan(_step, sequences=[mask, state_in], outputs_info=[state_out], name=_concat(prefix, '_layer'), n_steps=n_steps)
        return rval


def lstm(mask, state_in, t_params, n_dim_in, n_dim_out, prefix, one_step=False, state_out=None):
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

    if not t_params.has_key(_concat(prefix, 'b')):
        params = OrderedDict()
        params[_concat(prefix, 'W')] = numpy.concatenate([ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out)], axis=1)
        params[_concat(prefix, 'U')] = numpy.concatenate([ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out)], axis=1)
        params[_concat(prefix, 'b')] = numpy.zeros((4 * n_dim_out,)).astype(config.floatX)
        init_t_params(params, t_params)

    n_steps = state_in.shape[0]
    n_samples = state_in.shape[1] if state_in.ndim == 3 else 1
    state_in = (tensor.dot(state_in, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')])
    if state_out is None:
        state_out = tensor.alloc(to_t_float(0.), n_samples, n_dim_out)
    if one_step:
        rval, _ = _step(mask, state_in, state_out, tensor.alloc(to_t_float(0.), n_samples, n_dim_out))
    else:
        rval, _ = theano.scan(_step, sequences=[mask, state_in], outputs_info=[state_out, tensor.alloc(to_t_float(0.), n_samples, n_dim_out)], name=_concat(prefix, '_layer'), n_steps=n_steps)

    return rval[0]
