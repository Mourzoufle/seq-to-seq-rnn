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

    def __init__(self, max_items, max_size, header='', char='#'):
        self.__max_items = max_items    # Total number of items to be processed
        self.__max_size = max_size      # Max size of the progress bar - Total number of characters and spaces
        self.__header = header          # Header of the progress bar
        self.__char = char              # Character indicating the current progress
        self.__threshold = 1            # Threshold to trigger the progress bar to be redrawn
        self.__is_paused = True         # Flag indicating if the current progress is paused and has printed \n
        self.__is_finished = False      # Flag indicating if the current progress us already finished


    def disp(self, items, tag=''):
        '''
        Display the current progress according to the given items
        '''
        if self.__is_finished:
            return

        if not self.__is_paused and len(items) < self.__threshold:
            return

        self.__is_paused = False
        size = len(items) * self.__max_size / self.__max_items
        stdout.write('\r%s %3d%% |%s%s| %s: %f' % (self.__header, size * 100 / self.__max_size, self.__char * size, ' ' * (self.__max_size - size), tag, numpy.mean(items)))

        if len(items) == self.__max_items:
            self.__is_paused = True
            self.__is_finished = True
            stdout.write('\n')
            return

        self.__threshold = min(int((self.__max_items - 1) * float(size + 1) / self.__max_size + 1), self.__max_items)


    def pause(self):
        '''
        Pause the progress bar
        '''
        if not self.__is_paused:
            self.__is_paused = True
            stdout.write('\n')
