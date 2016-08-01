'''
Training process
'''

from __future__ import print_function
import time
from collections import OrderedDict

import numpy
from theano import tensor, config

from data import load_data, load_batch, get_batches
from layers import dense, embedding, gru
from optimizers import adadelta
from utils import init_t_params, params_zip, params_unzip, ProgressBar


def build_model(t_params, n_dim_img, n_dim_txt, n_dim_enc, n_dim_dec, n_dim_vocab, optimizer):
    '''
    Build the whole model for training
    '''
    x = tensor.tensor3('x', config.floatX)
    mask_x = tensor.matrix('mask_x', 'int8')
    # Encoder(s) and initialization of hidden layer
    enc = gru(mask_x, x, t_params, n_dim_img, n_dim_enc, 'enc')[-1]
    init_h = tensor.tanh(dense(enc, t_params, n_dim_enc, n_dim_dec, 'init_h'))

    y = tensor.matrix('y', 'int32')
    mask_y = tensor.matrix('mask_y', 'int8')
    n_steps, n_samples = y.shape
    # Repetition of the final state of hidden layer
    enc = tensor.tile(enc, (n_steps, 1, 1))
    # Word embedding
    emb = embedding(y, t_params, n_dim_vocab, n_dim_txt, 'emb').reshape((n_steps, n_samples, n_dim_txt))[: -1]
    emb = tensor.concatenate([tensor.zeros((1, n_samples, n_dim_txt), config.floatX), emb])
    # Decoder(s)
    dec = gru(mask_y, tensor.concatenate([enc, emb], 2), t_params, n_dim_enc + n_dim_txt, n_dim_dec, 'dec', init_h=init_h)
    # Full-connected layer
    fc = dense(tensor.concatenate([enc, emb, dec], 2), t_params, n_dim_enc + n_dim_txt + n_dim_dec, n_dim_vocab, 'fc')
    # Classifier
    prob = tensor.nnet.softmax(fc.reshape((n_steps * n_samples, n_dim_vocab)))
    # Cost function
    cost = prob[tensor.arange(n_steps * n_samples), y.flatten()].reshape((n_steps, n_samples))
    cost = ((-tensor.log(cost + 1e-6) * mask_y).sum(0) / mask_y.astype(config.floatX).sum(0)).mean()
    grads = tensor.grad(cost, list(t_params.values()))
    f_cost, f_update = optimizer(tensor.scalar('lr'), t_params, grads, [x, mask_x, y, mask_y], cost)

    return f_cost, f_update


def get_cost(f_cost, samples, batch, mat, costs, pgb, f_update=None, lrate=0.):
    '''
    Compute the cost of the given batches and update the parameters if it is necessary
    '''
    x, mask_x, y, mask_y = load_batch(samples, batch, mat)
    costs.extend(f_cost(x, mask_x, y, mask_y))
    if f_update:
        f_update(lrate)

    if numpy.isnan(costs[-1]) or numpy.isinf(costs[-1]):
        pgb.pause()
        raise FloatingPointError, 'Bad cost detected!'

    pgb.disp(costs, 'COST')


def main(
    # Dataset Configuration
    path_train='../train.json',             # Path to load training set
    path_val='../val.json',                 # Path to load validation set
    path_mat_train='../VGG19_train.npy',    # Path of image features of training set
    path_mat_val='../VGG19_val.npy',        # Path of image features of validation set
    max_samples_train=0,                    # Max number of samples in training set
    max_samples_val=0,                      # Max number of samples in validation set
    # Model Configuration
    n_dim_img=4096,                         # Dimension of image feature
    n_dim_txt=300,                          # Dimension of word embedding
    n_dim_enc=256,                          # Number of hidden units in encoder
    n_dim_dec=512,                          # Number of hidden units in decoder
    batch_size_train=64,                    # Batch size in training
    batch_size_test=64,                     # Batch size in validation
    optimizer=adadelta,                     # [sgd|adam|adadelta|rmsprop], sgd not recommanded
    lrate=0.0002,                           # Learning rate for optimizer
    max_epochs=1000,                        # Maximum number of epoch to run
    patience=10,                            # Number of epoch to wait before early stop if no progress
    # Frequency
    ratio_val=1.,                           # Validation frequency - Validate model after trained by this ratio of data
    ratio_save=1.,                          # Save frequency - Save the best parameters after trained by this ratio of data
    # Save & Load
    path_load=None,                         # Path to load a previouly trained model
    path_save='model',                      # Path to save the models
):
    '''
    Main function
    '''
    print('Loading data...')
    n_dim_vocab = 0                                             # Vocabulary size
    samples_train, mat_train, n_dim_vocab = load_data(path_train, path_mat_train, n_dim_vocab, max_samples_train)
    samples_val, mat_val, n_dim_vocab = load_data(path_val, path_mat_val, n_dim_vocab, max_samples_val)

    print('\ttraining:   %6d samples' % len(samples_train))
    print('\tvalidation: %6d samples' % len(samples_val))

    t_params = OrderedDict()
    best_params = None
    costs = []
    if path_load:
        best_params = OrderedDict(numpy.load(path_load))
        costs.extend(best_params['costs'])
        del best_params['costs']
        init_t_params(best_params, t_params)

    print('Building model...')
    f_cost, f_update = build_model(t_params, n_dim_img, n_dim_txt, n_dim_enc, n_dim_dec, n_dim_vocab, optimizer)

    print('Training...')
    time_start = time.time()
    batches_val = get_batches(len(samples_val), batch_size_test)
    n_epochs = 0
    n_samples = 0
    n_bad_costs = 0
    n_stops = 0
    next_val = ratio_val * len(samples_train)
    next_save = max(ratio_save * len(samples_train), next_val)
    while n_epochs < max_epochs:
        n_epochs += 1
        batches_train = get_batches(len(samples_train), batch_size_train, True)
        pgb_train = ProgressBar(len(batches_train), 20, 'EPOCH %4d ' % n_epochs)
        costs_train = []
        for batch_train in batches_train:
            n_samples += len(batch_train)
            get_cost(f_cost, samples_train, batch_train, mat_train, costs_train, pgb_train, f_update, lrate)

            if n_samples >= next_val:
                next_val += ratio_val * len(samples_train)
                pgb_train.pause()
                pgb_val = ProgressBar(len(batches_val), 20, 'VALIDATION ')
                costs_val = []
                for batch_val in batches_val:
                    get_cost(f_cost, samples_val, batch_val, mat_val, costs_val, pgb_val)
                costs.append(numpy.mean(costs_val))

                if best_params is None or costs[-1] <= numpy.min(costs):
                    best_params = params_unzip(t_params)
                    n_bad_costs = 0
                else:
                    n_bad_costs += 1
                    if n_bad_costs > patience:
                        n_stops += 1
                        print('WARNING: early stop for %d time(s)!' % n_stops)
                        params_zip(best_params, t_params)
                        n_bad_costs = 0

            if path_save and n_samples >= next_save:
                next_save = max(next_save + ratio_save * len(samples_train), next_val)
                pgb_train.pause()
                print('Saving model...')
                if best_params is not None:
                    params = best_params
                else:
                    params = params_unzip(t_params)
                numpy.savez(path_save, costs=costs, **params)
                numpy.savez('%s_%f' % (path_save, costs_train[-1]), costs=costs, **params_unzip(t_params))

    time_end = time.time()
    print('Training finished')
    print('TIME: %9.3f sec    EPOCHS: %4d    SPEED: %9.3f sec/epoch' % (time_end - time_start, n_epochs, (time_end - time_start) / n_epochs))

    if best_params is not None:
        params_zip(best_params, t_params)
    else:
        best_params = params_unzip(t_params)

    print('Saving final model...')
    if path_save:
        numpy.savez(path_save, costs=costs, **best_params)

    print('Done.')


if __name__ == '__main__':
    # For debugging, use the following arguments:
    # main(max_samples_train=400, max_samples_val=50, batch_size_train=50, batch_size_test=10, max_epochs=5, ratio_val=2)
    main(ratio_val=10, path_load='model.npz')
