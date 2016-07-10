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

from layers import dropout, dense, embedding, gru
from optimizers import rmsprop
from utils import to_floatX, init_t_params, params_zip, params_unzip


SEED = 65535
numpy.random.seed(SEED)
trng = MRG_RandomStreams(SEED)


def get_batches(n_samples, batch_size, shuffle=False):
    '''
    Get batches of data
    '''
    indices = numpy.arange(n_samples, dtype='int32')
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

    return numpy.swapaxes(to_floatX(x), 0, 1), numpy.asarray(mask_x, 'int8').T, numpy.asarray(y, 'int32').T, numpy.asarray(mask_y, 'int8').T


def build_model(t_params, n_dim_img, n_dim_txt, n_dim_enc, n_dim_dec, n_dim_vocab, optimizer):
    '''
    Build the whole model for training
    '''
    x = tensor.tensor3('x', config.floatX)
    mask_x = tensor.matrix('mask_x', 'int8')
    # Encoder(s)
    enc = gru(mask_x, x, t_params, n_dim_img, n_dim_enc, 'enc_1')[-1]
    # Initialization of hidden layer
    init_h = dense(enc, t_params, n_dim_enc, n_dim_dec, 'init_h')
    init_h = tensor.tanh(init_h)
    f_enc = theano.function([x, mask_x], [enc, init_h], name='f_enc')

    y = tensor.matrix('y', 'int32')
    mask_y = tensor.matrix('mask_y', 'int8')
    n_steps, n_samples = y.shape
    # Repetition of the final state of hidden layer
    enc = tensor.repeat(enc.dimshuffle('x', 0, 1), n_steps, 0)
    # Word embedding
    emb = embedding(y, t_params, n_dim_vocab, n_dim_txt, 'emb').reshape((n_steps, n_samples, n_dim_txt))[1:]
    emb = tensor.concatenate([tensor.zeros((1, n_samples, n_dim_txt)), emb])
    # Decoder(s)
    dec = gru(mask_y, tensor.concatenate([enc, emb], 2), t_params, n_dim_enc + n_dim_txt, n_dim_dec, 'dec_1', init_h=init_h)
    # Add dropout to the output of the final decoder
    dec = dropout(dec, to_floatX(1.), trng)
    # Classifier
    dec = dense(dec, t_params, n_dim_dec, n_dim_vocab, 'fc_1')
    dec = tensor.nnet.softmax(dec.reshape((n_steps * n_samples, n_dim_vocab)))
    # Cost function
    cost = dec[tensor.arange(n_steps * n_samples), y.flatten()].reshape((n_steps, n_samples))
    cost = ((-tensor.log(cost + 1e-6) * mask_y).sum(0) / mask_y.astype(config.floatX).sum(0)).mean()
    grads = tensor.grad(cost, list(t_params.values()))
    f_cost, f_update = optimizer(tensor.scalar('lr'), t_params, grads, [x, mask_x, y, mask_y], cost)

    return f_enc, f_cost, f_update


def build_sampler(t_params, n_dim_txt, n_dim_enc, n_dim_dec, n_dim_vocab):
    '''
    Build word sampler for validation / testing
    '''
    y = tensor.vector('y', 'int32')
    enc = tensor.matrix('enc', config.floatX)
    init_h = tensor.matrix('init_h', config.floatX)
    # Word embedding
    emb = tensor.switch(y[:, None] < 0, tensor.zeros((y.shape[0], n_dim_txt), config.floatX), embedding(y, t_params, n_dim_vocab, n_dim_txt, 'emb'))
    # Decoder(s) - Initialization of hidden layer in the next step
    next_h = gru(tensor.ones_like(y, 'int8'), tensor.concatenate([enc, emb], 1), t_params, n_dim_enc + n_dim_txt, n_dim_dec, 'dec_1', True, init_h)
    # Classifier
    dec = dense(0.5 * next_h, t_params, n_dim_dec, n_dim_vocab, 'fc_1')
    dec = tensor.nnet.softmax(dec)
    f_dec = theano.function([y, enc, init_h], [dec, next_h], name='f_dec')

    return f_dec


def get_err(f_enc, f_dec, samples, batches, mat, beam_size, max_len):
    '''
    Sample words using beam search and compute the prediction error
    '''
    preds = []
    errs = []
    for batch in batches:
        x, mask_x, y, mask_y = load_batch(samples, batch, mat)
        enc, init_h = f_enc(x, mask_x)

        n_samples = x.shape[1]
        hypo_sents = numpy.zeros((beam_size, n_samples, max_len), 'int32')
        hypo_scores = numpy.zeros((beam_size, n_samples), config.floatX)
        hypo_states = numpy.zeros((beam_size, n_samples, init_h.shape[-1]), config.floatX)

        # First step - No embedded word is fed into the decoder
        hypo_words = numpy.asarray([-1] * n_samples, 'int32')
        dec, init_h = f_dec(hypo_words, enc, init_h)
        for i in range(beam_size):
            words = dec.argmax(-1)
            hypo_sents[i, :, 0] = words
            hypo_scores[i] = numpy.log(dec[numpy.arange(n_samples), words] + 1e-6)
            dec[numpy.arange(n_samples), words] = 0
            hypo_states[i] = init_h

        for i in range(1, max_len - 1):
            sents = [[]] * n_samples
            scores = [[]] * n_samples
            states = [[]] * n_samples
            gen_hypos = numpy.asarray([False] * n_samples)
            for j in range(beam_size):
                hypo_words = hypo_sents[j, :, i - 1]
                dec, init_h = f_dec(hypo_words, enc, hypo_states[j])
                for k in range(n_samples):
                    if hypo_words[k] > 0:
                        gen_hypos[k] = True
                        for _ in range(beam_size):
                            word = dec[k].argmax()
                            hypo_sents[j, k, i] = word
                            sents[k].append(hypo_sents[j, k].copy())
                            scores[k].append(hypo_scores[j, k] + numpy.log(dec[k, word] + 1e-6))
                            states[k].append(init_h[k])
                            dec[k, word] = 0
                    else:
                        sents[k].append(hypo_sents[j, k].copy())
                        scores[k].append(hypo_scores[j, k])
                        states[k].append(hypo_states[j, k].copy())

            for j in range(n_samples):
                if not gen_hypos[j]:
                    continue
                for k in range(beam_size):
                    idx = numpy.argmax(scores[j])
                    hypo_sents[k, j] = sents[j][idx]
                    hypo_scores[k, j] = scores[j][idx]
                    hypo_states[k, j] = states[j][idx]
                    del sents[j][idx], scores[j][idx], states[j][idx]
            if not gen_hypos.any():
                break

        hypo_sents = hypo_sents[hypo_scores.argmax(0), numpy.arange(n_samples)]
        for i in range(n_samples):
            preds.append(hypo_sents[i, : (hypo_sents[i] > 0).sum() + 1].tolist())
        y = numpy.concatenate([y, numpy.zeros((max_len - len(y), n_samples), 'int32')]).T
        mask_y = numpy.concatenate([mask_y, numpy.zeros((max_len - len(mask_y), n_samples), 'int32')]).T
        errs.extend(((hypo_sents != y) * mask_y * 1.).sum(1) / mask_y.sum(1))

    return preds, numpy.mean(errs)


def main_model(
    # Dataset Configuration
    path_train='../train.json',             # Path to load training set
    path_val='../val.json',                 # Path to load validation set
    path_test='../test.json',               # Path to load testing set
    path_mat_train='../VGG19_train.npy',    # Path of image features of training set
    path_mat_val='../VGG19_val.npy',        # Path of image features of validation set
    path_mat_test='../VGG19_test.npy',      # Path of image features of testing set
    max_samples_train=0,                    # Max number of samples in training
    max_samples_val=0,                      # Max number of samples in validating
    max_samples_test=0,                     # Max number of samples in testing
    # Model Configuration
    n_dim_img=4096,                         # Dimension of image feature
    n_dim_txt=300,                          # Dimension of word embedding
    n_dim_enc=256,                          # Number of hidden units in encoder
    n_dim_dec=512,                          # Number of hidden units in decoder
    batch_size_train=64,                    # Batch size in training
    batch_size_test=256,                    # Batch size in validation / testing
    optimizer=rmsprop,                      # [sgd|adam|adadelta|rmsprop], sgd not recommanded
    lrate=0.0002,                           # Learning rate for optimizer
    max_epochs=1000,                        # Maximum number of epoch to run
    patience=20,                            # Number of epoch to wait before early stop if no progress
    max_err_valid=0.75,                     # Max accepted error in validation, error above this threshold will cause NO early stop
    beam_size=10,                           # number of candidate(s) in beam search
    # Save & Load
    path_load=None,                         # Path to load a previouly trained model
    path_save='model.npy',                  # Path to save the best model
    path_out_train='out_train.json',        # Path to save predicted sentences of training set
    path_out_val='out_val.json',            # Path to save predicted sentences of validation set
    path_out_test='out_test.json',          # Path to save predicted sentences of testing set
):
    '''
    The main process to do visual storytelling via a seq-to-seq RNN model
    '''
    print('Loading data...')
    n_dim_vocab = 0                                             # Vocabulary size
    samples_train, mat_train, n_dim_vocab = load_data(path_train, path_mat_train, n_dim_vocab)
    if max_samples_train > 0:
        samples_train = samples_train[: max_samples_train]
    max_len = max([len(sample[1]) for sample in samples_train]) # Max length of sentences

    samples_val, mat_val, n_dim_vocab = load_data(path_val, path_mat_val, n_dim_vocab)
    if max_samples_val > 0:
        samples_val = samples_val[: max_samples_val]

    samples_test, mat_test, n_dim_vocab = load_data(path_test, path_mat_test, n_dim_vocab)
    if max_samples_test > 0:
        samples_test = samples_test[: max_samples_test]

    batches_val = get_batches(len(samples_val), batch_size_test)
    batches_test = get_batches(len(samples_test), batch_size_test)
    freq_disp = ((len(samples_train) - 1) / batch_size_train) / 20 + 1
    freq_val = ((len(samples_train) - 1) / batch_size_train) / 4 + 1
    print('\ttraining:   %6d samples' % len(samples_train))
    print('\tvalidation: %6d samples' % len(samples_val))
    print('\ttesting:    %6d samples' % len(samples_test))

    t_params = OrderedDict()
    if path_load:
        init_t_params(numpy.load(path_load), t_params)
    print('Building model...')
    f_enc, f_cost, f_update = build_model(t_params, n_dim_img, n_dim_txt, n_dim_enc, n_dim_dec, n_dim_vocab, optimizer)
    print('Building word sampler...')
    f_dec = build_sampler(t_params, n_dim_txt, n_dim_enc, n_dim_dec, n_dim_vocab)

    print('Training...')
    time_start = time.time()
    history_errs = []
    best_p = None
    bad_count = 0
    stop = False
    n_epochs = 0
    n_batches = 0
    while n_epochs < max_epochs:
        n_epochs += 1
        batches_train = get_batches(len(samples_train), batch_size_train, True)
        costs = []
        for idx_batch, batch in enumerate(batches_train):
            n_batches += 1
            idx_batch += 1
            x, mask_x, y, mask_y = load_batch(samples_train, batch, mat_train)
            costs.append(f_cost(x, mask_x, y, mask_y))
            f_update(lrate)

            if numpy.isnan(costs[-1]) or numpy.isinf(costs[-1]):
                print('ERROR: bad cost detected!')
                return

            if numpy.mod(idx_batch, freq_disp) == 0:
                print('\tEPOCH: %6d    BATCH: %6d    COST: %12.6f' % (n_epochs, n_batches, numpy.mean(costs)))

            if numpy.mod(idx_batch, freq_val) == 0:
                print('Validating...')
                _, err_train = get_err(f_enc, f_dec, samples_train, batches_train[idx_batch - freq_val: idx_batch], mat_train, beam_size, max_len)
                _, err_val = get_err(f_enc, f_dec, samples_val, batches_val, mat_val, beam_size, max_len)
                _, err_test = get_err(f_enc, f_dec, samples_test, batches_test, mat_test, beam_size, max_len)
                print('ERR_TRA: %.6f    ERR_VAL: %.6f    ERR_TES: %.6f' % (err_train, err_val, err_test))

                history_errs.append([err_val, err_test])
                if best_p is None or err_val <= numpy.array(history_errs)[:, 0].min():
                    best_p = params_unzip(t_params)
                    bad_count = 0
                elif n_epochs > 1:
                    bad_count += 1
                    if bad_count > patience and err_val < max_err_valid:
                        print('WARNING: early stop!')
                        stop = True
                        break

        if path_save:
            print('Saving model...')
            if best_p is not None:
                params = best_p
            else:
                params = params_unzip(t_params)
            numpy.save(path_save, params)

        if stop:
            break

    time_end = time.time()
    print('Training finished')
    print('TIME: %9.3f sec    EPOCHS: %6d    SPEED: %9.3f sec/epoch' % (time_end - time_start, n_epochs, (time_end - time_start) / n_epochs))

    if best_p is not None:
        params_zip(best_p, t_params)
    else:
        best_p = params_unzip(t_params)

    print('Final predicting...')
    preds_train, err_train = get_err(f_enc, f_dec, samples_train, get_batches(len(samples_train), batch_size_test), mat_train, beam_size, max_len)
    preds_val, err_val = get_err(f_enc, f_dec, samples_val, batches_val, mat_val, beam_size, max_len)
    preds_test, err_test = get_err(f_enc, f_dec, samples_test, batches_test, mat_test, beam_size, max_len)
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
    # For debugging, use the following arguments:
    # main_model(max_samples_train=400, max_samples_val=50, max_samples_test=50, batch_size_train=50, batch_size_test=50, max_epochs=10, beam_size=2, path_save=None)
    main_model()
