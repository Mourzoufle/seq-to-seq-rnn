'''
A sequence-to-sequence recurrent neural network model for visual storytelling
'''

from __future__ import print_function
import copy
import time
import json
from collections import OrderedDict

import numpy
import theano
from theano import tensor, config
from theano.sandbox.rng_mrg import MRG_RandomStreams

from layers import dropout, dense, gru
from optimizers import rmsprop
from utils import to_t_float, init_t_params, params_zip, params_unzip


SEED = 65535


def get_batches(n_samples, batch_size, shuffle=False):
    '''
    Get batches of data
    '''
    indices = numpy.arange(n_samples, dtype="int32")
    if shuffle:
        numpy.random.shuffle(indices)
    batches = []
    idx_start = 0
    while idx_start < n_samples:
        batches.append(indices[idx_start: idx_start + batch_size])
        idx_start += batch_size

    return batches


def load_data(path_list, path_mat, n_dim_y):
    '''
    Load dataset
    '''
    with open(path_list, 'r') as file_in:
        samples = json.load(file_in)
        for i, _ in enumerate(samples):
            samples[i][0] = samples[i][0][: : -1]
            n_dim_y = max(n_dim_y, max(samples[i][1]) + 1)

    return samples, numpy.load(path_mat), n_dim_y


def load_batch(samples, batch, mat):
    '''
    Fetch and expand data of a batch in NumPy ndarray format
    '''
    x = []
    y = []
    for idx in batch:
        x.append(copy.deepcopy(samples[idx][0]))
        y.append(copy.deepcopy(samples[idx][1]))
    max_len_x = max(len(row) for row in x)
    max_len_y = max(len(row) for row in y)

    mask_x = []
    mask_y = []
    n_samples = len(batch)
    for i in range(n_samples):
        mask_x.append([1] * len(x[i]))
        x[i].extend([0] * (max_len_x - len(x[i])))
        mask_x[i].extend([0] * (max_len_x - len(mask_x[i])))
        mask_y.append([1] * len(y[i]))
        y[i].extend([0] * (max_len_y - len(y[i])))
        mask_y[i].extend([0] * (max_len_y - len(mask_y[i])))
    x = mat[numpy.asarray(x).flatten()].reshape(n_samples, max_len_x, mat.shape[-1])

    return numpy.swapaxes(numpy.array(x, dtype=config.floatX), 0, 1), numpy.array(mask_x, dtype='int8').T, numpy.array(y, dtype='int64').T, numpy.array(mask_y, dtype='int8').T


def pred_error(f_prob, samples, batches, mat):
    '''
    Compute the prediction error
    '''
    preds = []
    errs = []
    for batch in batches:
        x, mask_x, y, mask_y = load_batch(samples, batch, mat)
        pred = f_prob(x, mask_x, y, mask_y).argmax(axis=-1)
        for i in range(1, pred.shape[0]):
            pred[i] = pred[i] * (pred[i - 1] > 0)
        preds.extend(pred.T.tolist())
        mask_y = mask_y + pred > 0
        errs.extend((((pred != y) * mask_y).sum(axis=0).astype(config.floatX) / mask_y.sum(axis=0)).tolist())

    return preds, numpy.mean(errs)


def main_model(
    # Dataset Configuration
    path_train='train.json',            # Path to load training set
    path_val='val.json',                # Path to load validation set
    path_test='test.json',              # Path to load testing set
    path_mat_train='VGG19_train.npy',   # Path of image features of training set
    path_mat_val='VGG19_val.npy',       # Path of image features of validation set
    path_mat_test='VGG19_test.npy',     # Path of image features of testing set
    max_samples_train=0,                # Max number of samples in training
    max_samples_val=0,                  # Max number of samples in validating
    max_samples_test=0,                 # Max number of samples in testing
    # Model Configuration
    n_dim_img=4096,                     # Image feature dimension
    n_dim_txt=0,                        # Vocabulary size
    n_dim_enc=256,                      # Number of hidden units in encoder
    n_dim_dec=256,                      # Number of hidden units in decoder
    batch_size_train=64,                # Batch size in training
    batch_size_pred=256,                # Batch size in validation / testing
    optimizer=rmsprop,                  # [sgd|adadelta|rmsprop], sgd not recommanded
    lrate=0.0001,                       # Learning rate for optimizer
    max_epochs=1000,                    # Maximum number of epoch to run
    patience=10,                        # Number of epoch to wait before early stop if no progress
    # Save & Load
    path_load=None,                     # Path to load a previouly trained model
    path_save='model.npy',              # Path to save the best model
    path_out_train='out_train.json',    # Path to save predicted sentences of training set
    path_out_val='out_val.json',        # Path to save predicted sentences of validation set
    path_out_test='out_test.json',      # Path to save predicted sentences of testing set
    # MISC
    freq_disp=20,                       # Display the training progress after this number of updates
    freq_val=200,                       # Compute the validation error after this number of updates
    freq_save=1000,                     # Save the parameters after this number of updates
):
    '''
    The main process to do visual storytelling via a seq-to-seq RNN model
    '''
    print('Loading data...')
    samples_train, mat_train, n_dim_txt = load_data(path_train, path_mat_train, n_dim_txt)
    samples_val, mat_val, n_dim_txt = load_data(path_val, path_mat_val, n_dim_txt)
    samples_test, mat_test, n_dim_txt = load_data(path_test, path_mat_test, n_dim_txt)
    if max_samples_train > 0:
        samples_train = samples_train[: max_samples_train]
    if max_samples_val > 0:
        samples_val = samples_val[: max_samples_val]
    if max_samples_test > 0:
        samples_test = samples_test[: max_samples_test]
    print('\ttraining:   %6d samples' % len(samples_train))
    print('\tvalidation: %6d samples' % len(samples_val))
    print('\ttesting:    %6d samples' % len(samples_test))

    print('Building model...')
    numpy.random.seed(SEED)
    trng = MRG_RandomStreams(SEED)
    # Flag of dropout
    use_dropout = theano.shared(to_t_float(0.))
    # Inputs
    x = tensor.tensor3('x', dtype=config.floatX)
    mask_x = tensor.matrix('mask_x', dtype='int8')
    y = tensor.matrix('y', dtype='int64')
    mask_y = tensor.matrix('mask_y', dtype='int8')

    n_steps, n_samples = y.shape
    t_params = OrderedDict()
    if path_load:
        init_t_params(numpy.load(path_load), t_params)
    # Encoder(s)
    out_enc = gru(mask_x, x, t_params, n_dim_img, n_dim_enc, 'enc_1')
    # Repetition of the final state of hidden layer
    out_enc = tensor.repeat(out_enc[-1].dimshuffle('x', 0, 1), n_steps, axis=0)
    # Decoder(s)
    out_dec = gru(mask_y, out_enc, t_params, n_dim_enc, n_dim_dec, 'dec_1')
    # Add dropout to the output of the final decoder
    out_dec = dropout(out_dec, use_dropout, trng)
    # Classifier
    out_fc = dense(out_dec, t_params, n_dim_dec, n_dim_txt, 'fc_1')
    out_fc = tensor.nnet.softmax(out_fc.reshape((n_steps * n_samples, n_dim_txt)))
    # Cost function
    offset = 1e-8 if out_fc.dtype == 'float16' else 1e-6
    cost = out_fc[tensor.arange(n_steps * n_samples), y.flatten()].reshape((n_steps, n_samples))
    cost = (-tensor.log(cost + offset) * mask_y).sum() / mask_y.sum()
    grads = tensor.grad(cost, wrt=list(t_params.values()))
    f_prob = theano.function([x, mask_x, y, mask_y], out_fc.reshape((n_steps, n_samples, n_dim_txt)), name='f_prob')
    f_grad_shared, f_update = optimizer(tensor.scalar(name='lr'), t_params, grads, x, mask_x, y, mask_y, cost)

    print('Training...')
    batches_val = get_batches(len(samples_val), batch_size_pred)
    batches_test = get_batches(len(samples_test), batch_size_pred)
    history_errs = []
    best_p = None
    bad_count = 0
    if freq_val <= 0:
        freq_val = (len(samples_train) - 1) / batch_size_train + 1
    if freq_save <= 0:
        freq_save = ((len(samples_train) - 1) / batch_size_train + 1) * max_epochs + 1 # No automatic saving
    n_epoch = 0
    n_batches = 0
    stop = False
    time_start = time.time()

    while n_epoch < max_epochs:
        n_epoch += 1
        batches_train = get_batches(len(samples_train), batch_size_train, shuffle=True)
        for batch in batches_train:
            n_batches += 1
            use_dropout.set_value(1.)
            x, mask_x, y, mask_y = load_batch(samples_train, batch, mat_train)
            cost = f_grad_shared(x, mask_x, y, mask_y)
            f_update(lrate)

            if numpy.isnan(cost) or numpy.isinf(cost):
                print('ERROR: bad cost detected!')
                return

            if numpy.mod(n_batches, freq_disp) == 0:
                print('\tEPOCH: %6d    BATCH: %6d    COST: %12.6f' % (n_epoch, n_batches, cost))

            if path_save and numpy.mod(n_batches, freq_save) == 0:
                print('Saving model...')
                if best_p is not None:
                    params = best_p
                else:
                    params = params_unzip(t_params)
                numpy.save(path_save, params)

            if numpy.mod(n_batches, freq_val) == 0:
                print('Validating...')
                use_dropout.set_value(0.)
                _, err_train = pred_error(f_prob, samples_train, batches_train, mat_train)
                _, err_val = pred_error(f_prob, samples_val, batches_val, mat_val)
                _, err_test = pred_error(f_prob, samples_test, batches_test, mat_test)
                history_errs.append([err_val, err_test])

                if (best_p is None or err_val <= numpy.array(history_errs)[:, 0].min()):
                    best_p = params_unzip(t_params)
                    bad_count = 0
                print('ERR_TRA: %.6f    ERR_VAL: %.6f    ERR_TES: %.6f' % (err_train, err_val, err_test))
                if len(history_errs) > patience and err_val >= numpy.array(history_errs)[: -patience, 0].min():
                    bad_count += 1
                    if bad_count > patience:
                        print('WARNING: early stop!')
                        stop = True
                        break
        if stop:
            break
    time_end = time.time()
    print('Training finished')
    print('TIME: %9.3f sec    EPOCHS: %6d    SPEED: %9.3f sec/epoch' % (time_end - time_start, n_epoch, (time_end - time_start) / n_epoch))

    if best_p is not None:
        params_zip(best_p, t_params)
    else:
        best_p = params_unzip(t_params)

    print('Final predicting...')
    use_dropout.set_value(0.)
    preds_train, err_train = pred_error(f_prob, samples_train, get_batches(len(samples_train), batch_size_pred), mat_train)
    preds_val, err_val = pred_error(f_prob, samples_val, batches_val, mat_val)
    preds_test, err_test = pred_error(f_prob, samples_test, batches_test, mat_test)
    print('ERR_TRA: %.6f    ERR_VAL: %.6f    ERR_TES: %.6f' % (err_train, err_val, err_test))
    if path_save:
        numpy.save(path_save, best_p)

    with open(path_out_train, 'w') as file_out:
        json.dump(preds_train, file_out)
    with open(path_out_val, 'w') as file_out:
        json.dump(preds_val, file_out)
    with open(path_out_test, 'w') as file_out:
        json.dump(preds_test, file_out)
    print('All works finished')


if __name__ == '__main__':
    main_model()
    # For debugging, use the following args:
    # main_model(max_samples_train=400, max_samples_val=50, max_samples_test=50, batch_size_train=50, batch_size_pred=50, max_epochs=10, freq_disp=1, freq_val=0, freq_save=0)
