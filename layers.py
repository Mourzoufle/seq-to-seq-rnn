'''
Neural network layers
'''

from collections import OrderedDict

import numpy
import theano
from theano import tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams

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
    u, _, v = numpy.linalg.svd(scale * numpy.random.randn(n_dim_in, n_dim_out).astype(config.floatX), False)
    if n_dim_in >= n_dim_out:
        return u
    else:
        return v


def dropout(state_in, use_dropout, trng=None, ratio=0.5):
    '''
    Dropout layer
    '''
    if trng is None:
        trng = MRG_RandomStreams()

    return tensor.switch(use_dropout, state_in * trng.binomial(state_in.shape, p=1 - ratio, dtype=state_in.dtype), state_in * (1 - ratio))


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
    def _step(_m, _x, _h):
        pre_act = tensor.dot(_h, t_params[_concat(prefix, 'U')])

        gate_r = tensor.nnet.sigmoid(_slice(pre_act, 0, n_dim_out) + _slice(_x, 0, n_dim_out))
        gate_u = tensor.nnet.sigmoid(_slice(pre_act, 1, n_dim_out) + _slice(_x, 1, n_dim_out))

        _h_new = tensor.tanh(gate_r * _slice(pre_act, 2, n_dim_out) + _slice(_x, 2, n_dim_out))
        _h_new = (1. - gate_u) * _h + gate_u * _h_new
        _h_new = _m[:, None] * _h_new + (1. - _m)[:, None] * _h

        return _h_new

    if _concat(prefix, 'W') not in t_params:
        params = OrderedDict()
        params[_concat(prefix, 'W')] = numpy.concatenate([ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out)], 1)
        params[_concat(prefix, 'U')] = numpy.concatenate([ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out)], 1)
        params[_concat(prefix, 'b')] = numpy.zeros((3 * n_dim_out,), config.floatX)
        init_t_params(params, t_params)

    if one_step:
        n_steps = 1
        n_samples = state_in.shape[0]
    else:
        n_steps, n_samples, _ = state_in.shape
    state_in = (tensor.dot(state_in, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')])
    if init_h is None:
        init_h = tensor.alloc(to_t_float(0.), n_samples, n_dim_out)
    if one_step:
        return _step(mask, state_in, init_h)
    else:
        rval, _ = theano.scan(_step, [mask, state_in], init_h, n_steps=n_steps, name=_concat(prefix, '_scan'))
        return rval


def lstm(mask, state_in, t_params, n_dim_in, n_dim_out, prefix, one_step=False, init_h=None):
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

    if _concat(prefix, 'W') not in t_params:
        params = OrderedDict()
        params[_concat(prefix, 'W')] = numpy.concatenate([ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out), ortho_weight(n_dim_in, n_dim_out)], 1)
        params[_concat(prefix, 'U')] = numpy.concatenate([ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out), ortho_weight(n_dim_out, n_dim_out)], 1)
        params[_concat(prefix, 'b')] = numpy.zeros((4 * n_dim_out,), config.floatX)
        init_t_params(params, t_params)

    if one_step:
        n_steps = 1
        n_samples = state_in.shape[0]
    else:
        n_steps, n_samples, _ = state_in.shape
    state_in = (tensor.dot(state_in, t_params[_concat(prefix, 'W')]) + t_params[_concat(prefix, 'b')])
    if init_h is None:
        init_h = tensor.alloc(to_t_float(0.), n_samples, n_dim_out)
    if one_step:
        rval, _ = _step(mask, state_in, init_h, tensor.alloc(to_t_float(0.), n_samples, n_dim_out))
        return rval
    else:
        rval, _ = theano.scan(_step, [mask, state_in], [init_h, tensor.alloc(to_t_float(0.), n_samples, n_dim_out)], n_steps=n_steps, name=_concat(prefix, '_scan'))
        return rval[0]
