'''
Optimizers
'''

import theano
from theano import tensor

from utils import to_floatX


def sgd(lr, t_params, grads, inputs, cost):
    '''
    Stochastic Gradient Descent

    Parameters
    ----------
    lr: Theano SharedVariable variable
        Initial learning rate
    t_params: Theano SharedVariable variable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t. to parameres
    inputs: list of Theano variables
        Inputs
    cost: Theano variable
        Objective fucntion to minimize

    NOTE: A more complicated version of sgd then needed. This is done like that for adadelta and rmsprop.
    '''
    # New SharedVariable set that will contain the gradient for a mini-batch
    gshared = [theano.shared(p.get_value() * to_floatX(0.), '%s_grad' % k) for k, p in t_params.iteritems()]

    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    # Function that computes gradients for a mini-batch, but do not updates the weights
    f_cost = theano.function(inputs, cost, updates=gsup, name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(t_params.values(), gshared)]
    # Function that updates the weights from the previously computed gradient
    f_update = theano.function([lr], [], updates=pup, name='sgd_f_update')

    return f_cost, f_update


def adam(lr, t_params, grads, inputs, cost):
    '''
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr: Theano SharedVariable variable
        Initial learning rate
    t_params: Theano SharedVariable variable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t. to parameres
    inputs: list of Theano variables
        Inputs
    cost: Theano variable
        Objective fucntion to minimize

    NOTE: For more information, see Diederik Kingma, Jimmy Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980
    '''
    gshared = [theano.shared(p.get_value() * to_floatX(0.), '%s_grad' % k) for k, p in t_params.iteritems()]
    m_up = [theano.shared(p.get_value() * to_floatX(0.), '%s_mup' % k) for k, p in t_params.iteritems()]
    v_up = [theano.shared(p.get_value() * to_floatX(0.), '%s_vup' % k) for k, p in t_params.iteritems()]

    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_cost = theano.function(inputs, cost, updates=gsup, name='adam_f_grad_shared')

    e = theano.shared(to_floatX(0.), 'adam_e')
    lr *= tensor.sqrt(1 - 0.001 ** (e + 1)) / (1 - 0.1 ** (e + 1))
    m_up2 = [(m, 0.1 * g + 0.9 * m) for m, g in zip(m_up, gshared)]
    v_up2 = [(v, 0.001 * tensor.sqrt(g) + 0.999 * v) for v, g in zip(v_up, gshared)]
    param_up = [(p, p - lr * m / (tensor.sqrt(v) + 1e-8)) for m, v, p, g in zip(m_up2, v_up2, t_params.values(), gshared)]
    f_update = theano.function([lr], [], updates=m_up2 + v_up2 + param_up + (e, e + 1), on_unused_input='ignore', name='adam_f_update')

    return f_cost, f_update


def adadelta(lr, t_params, grads, inputs, cost):
    '''
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr: Theano SharedVariable variable
        Initial learning rate
    t_params: Theano SharedVariable variable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t. to parameres
    inputs: list of Theano variables
        Inputs
    cost: Theano variable
        Objective fucntion to minimize

    NOTE: For more information, see Matthew D. Zeiler. ADADELTA: An Adaptive Learning Rate Method. arXiv:1212.5701.
    '''

    zipped_grads = [theano.shared(p.get_value() * to_floatX(0.), '%s_grad' % k) for k, p in t_params.iteritems()]
    running_up2 = [theano.shared(p.get_value() * to_floatX(0.), '%s_rup2' % k) for k, p in t_params.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * to_floatX(0.), '%s_rgrad2' % k) for k, p in t_params.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]
    f_cost = theano.function(inputs, cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(t_params.values(), updir)]
    f_update = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

    return f_cost, f_update


def rmsprop(lr, t_params, grads, inputs, cost):
    '''
    A variant of SGD that scales the step size by running average of the recent step norms.

    Parameters
    ----------
    lr: Theano SharedVariable variable
        Initial learning rate
    t_params: Theano SharedVariable variable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t. to parameres
    inputs: list of Theano variables
        Inputs
    cost: Theano variable
        Objective fucntion to minimize

    NOTE: For more information, see Geoff Hinton. Neural Networks for Machine Learning, lecture 6a. http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
    '''

    zipped_grads = [theano.shared(p.get_value() * to_floatX(0.), '%s_grad' % k) for k, p in t_params.iteritems()]
    running_grads = [theano.shared(p.get_value() * to_floatX(0.), '%s_rgrad' % k) for k, p in t_params.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * to_floatX(0.), '%s_rgrad2' % k) for k, p in t_params.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]
    f_cost = theano.function(inputs, cost, updates=zgup + rgup + rg2up, name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * to_floatX(0.), '%s_updir' % k) for k, p in t_params.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(t_params.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up, on_unused_input='ignore', name='rmsprop_f_update')

    return f_cost, f_update
