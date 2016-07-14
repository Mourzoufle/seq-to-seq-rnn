'''
Useful functions for data transfer between CPU and GPU
'''

from sys import stdout
from collections import OrderedDict

import numpy
import theano
from theano import config


def to_floatX(data):
    '''
    Convert given data to float format used by Theano
    '''
    return numpy.asarray(data, config.floatX)


def init_t_params(params, t_params):
    '''
    Initialize Theano parameters from given values
    '''
    for key, value in params.items():
        t_params[key] = theano.shared(value, key)


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


class ProgressBar(object):
    '''
    A General progress bar to display information
    '''

    def __init__(self, max_items, max_size, header='', char='>'):
        self.max_items = max_items  # Total number of items to be processed
        self.max_size = max_size    # Max size of the progress bar - Total number of characters and spaces
        self.header = header        # Header of the progress bar
        self.char = char            # Character indicating the current progress
        self.threshold = 1          # Threshold to trigger the progress bar to be redrawn


    def disp(self, items, tag=''):
        '''
        Display the current progress according to the given items
        '''
        if len(items) < self.threshold:
            return
        size = len(items) * self.max_size / self.max_items
        stdout.write('\r%s\t%6d%% [%s%s] %s:\t%f' % (self.header, size * 100 / self.max_size, self.char * size, ' ' * (self.max_size - size), tag, numpy.mean(items)))
        if len(items) == self.max_items:
            stdout.write('\n')
            return
        self.threshold = min(self.threshold + int(self.max_items * 1. / self.max_size + 0.5), self.max_items)
