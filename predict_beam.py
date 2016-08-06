'''
Prediction process - Beam Search
'''

from __future__ import print_function
import json
from collections import OrderedDict

import numpy
import theano
from theano import tensor, config

from data import load_data, load_batch, get_batches
from layers import dense, embedding, gru
from utils import init_t_params, ProgressBar


def build_enc(t_params, n_dim_img, n_dim_enc, n_dim_dec):
    '''
    Build the encoder for images
    '''
    x = tensor.tensor3('x', config.floatX)
    mask_x = tensor.matrix('mask_x', 'int8')
    # Encoder(s) and initialization of hidden layer
    enc = gru(mask_x, 0.5 * x, t_params, n_dim_img, n_dim_enc, 'enc')[-1]
    init_h = tensor.tanh(dense(enc, t_params, n_dim_enc, n_dim_dec, 'init_h'))

    return theano.function([x, mask_x], [init_h], name='f_enc')


def build_dec(t_params, n_dim_txt, n_dim_enc, n_dim_dec, n_dim_vocab, beam_size):
    '''
    Build the decoder for texts
    '''
    def _step(_prob):
        _y = _prob.argmax(-1)
        _log_prob = tensor.log(_prob[tensor.arange(_y.shape[0]), _y] + 1e-6)
        tensor.set_subtensor(_prob[tensor.arange(_y.shape[0]), _y], 0)

        return _y, _log_prob

    y = tensor.vector('y', 'int32')
    init_h = tensor.matrix('init_h', config.floatX)
    n_samples = y.shape[0]
    # Word embedding
    emb = tensor.switch(y[:, None] < 0, tensor.zeros((n_samples, n_dim_txt), config.floatX), embedding(y, t_params, n_dim_vocab, n_dim_txt, 'emb'))
    # Decoder(s) - Initialization of hidden layer in the next step
    next_h = gru(tensor.ones_like(y, 'int8'), emb, t_params, n_dim_txt, n_dim_dec, 'dec', True, init_h)
    # Full-connected layer
    fc = dense(0.5 * next_h, t_params, n_dim_dec, n_dim_vocab, 'fc')
    # Classifier
    prob = tensor.nnet.softmax(fc)
    # Hypo words
    [next_y, next_log_prob], _ = theano.scan(_step, non_sequences=prob, n_steps=beam_size)

    return theano.function([y, init_h], [next_y, next_log_prob, next_h], name='f_dec')


def predict(f_enc, f_dec, samples, batches, mat, beam_size, max_len, header):
    '''
    Sample words and compute the prediction error
    '''
    preds = []
    errs = []
    progress = ProgressBar(numpy.sum([len(batch) for batch in batches]), 20, header)
    for batch in batches:
        x, mask_x, y, mask_y = load_batch(samples, batch, mat)
        [init_h] = f_enc(x, mask_x)

        n_steps = mask_x.sum(0)
        n_samples = x.shape[1]
        prev_sents = numpy.zeros((beam_size, n_samples, max_len), 'int32')
        # First step - No embedded word is fed into the decoder
        prev_words = numpy.asarray([-1] * n_samples, 'int32')
        prev_sents[:, :, 0], prev_log_prob, prev_h = f_dec(prev_words, init_h)
        prev_h = numpy.tile(prev_h, (beam_size, 1, 1))
        prev_n_ends = n_steps - (prev_sents[:, :, 0] == 0)

        for i in range(1, max_len - 1):
            hypo_sents = [[]] * n_samples
            hypo_log_prob = [[]] * n_samples
            hypo_h = [[]] * n_samples
            hypo_n_ends = [[]] * n_samples
            has_hypos = numpy.asarray([False] * n_samples)
            for j in range(beam_size):
                if not prev_n_ends[j].any():
                    continue

                next_words, next_log_prob, next_h = f_dec(prev_sents[j, :, i - 1], prev_h[j])
                for k in range(n_samples):
                    if prev_n_ends[j, k] > 0:
                        has_hypos[k] = True
                        next_sents = numpy.tile(prev_sents[j, k], (beam_size, 1))
                        next_sents[:, i] = next_words[:, k]
                        hypo_sents[k].extend(next_sents)
                        hypo_log_prob[k].extend(next_log_prob[:, k] + prev_log_prob[j, k])
                        hypo_h[k].extend([next_h[k]] * beam_size)
                        hypo_n_ends[k].extend(prev_n_ends[j, k] - (next_words[:, k] == 0))
                    else:
                        hypo_sents[k].append(prev_sents[j, k].copy())
                        hypo_log_prob[k].append(prev_log_prob[j, k])
                        hypo_h[k].append(prev_h[j, k].copy())
                        hypo_n_ends[k].append(0)

            if not has_hypos.any():
                break

            for j in range(n_samples):
                if not has_hypos[j]:
                    continue

                indices = numpy.argsort(hypo_log_prob[j])[: -beam_size - 1: -1]
                for k in range(beam_size):
                    prev_sents[k, j] = hypo_sents[j][indices[k]]
                    prev_log_prob[k, j] = hypo_log_prob[j][indices[k]]
                    prev_h[k, j] = hypo_h[j][indices[k]]
                    prev_n_ends[k, j] = hypo_n_ends[j][indices[k]]

        sents = prev_sents[prev_log_prob.argmax(0), numpy.arange(n_samples)]
        for i in range(n_samples):
            idx = 0
            while idx < max_len and n_steps[i] > 0:
                if sents[i, idx] == 0:
                    n_steps[i] -= 1
                idx += 1
            preds.append(sents[i, : idx].tolist())

        y = numpy.concatenate([y, numpy.zeros((max_len - len(y), n_samples), 'int32')]).T
        mask_y = numpy.concatenate([mask_y, numpy.zeros((max_len - len(mask_y), n_samples), 'int32')]).T
        errs.extend(((sents != y) * mask_y * 1.).sum(1) / mask_y.sum(1))
        progress.disp(errs, ' ERR')

    return preds, numpy.mean(errs)


def main(
    # Dataset Configuration
    path_train='../train.json',             # Path to load training set
    path_val='../val.json',                 # Path to load validation set
    path_test='../test.json',               # Path to load testing set
    path_mat_train='../VGG19_train.npy',    # Path of image features of training set
    path_mat_val='../VGG19_val.npy',        # Path of image features of validation set
    path_mat_test='../VGG19_test.npy',      # Path of image features of testing set
    max_samples_train=0,                    # Max number of samples in training set
    max_samples_val=0,                      # Max number of samples in validation set
    max_samples_test=0,                     # Max number of samples in testing set
    # Model Configuration
    n_dim_img=4096,                         # Dimension of image feature
    n_dim_txt=250,                          # Dimension of word embedding
    n_dim_enc=1000,                         # Number of hidden units in encoder
    n_dim_dec=1000,                         # Number of hidden units in decoder
    batch_size=64,                          # Batch size
    beam_size=10,                           # number of candidate(s) in beam search
    # Save & Load
    path_load='model.npz',                  # Path to load a previouly trained model - Required
    path_out_train='beam_train.json',       # Path to save predicted sentences of training set
    path_out_val='beam_val.json',           # Path to save predicted sentences of validation set
    path_out_test='beam_test.json',         # Path to save predicted sentences of testing set
):
    '''
    Main function
    '''
    print('Loading data...')
    n_dim_vocab = 0                                             # Vocabulary size
    samples_train, mat_train, n_dim_vocab = load_data(path_train, path_mat_train, n_dim_vocab, max_samples_train)
    samples_val, mat_val, n_dim_vocab = load_data(path_val, path_mat_val, n_dim_vocab, max_samples_val)
    samples_test, mat_test, n_dim_vocab = load_data(path_test, path_mat_test, n_dim_vocab, max_samples_test)
    max_len = max([len(sample[1]) for sample in samples_train]) # Max length of sentences

    print('\ttraining:   %6d samples' % len(samples_train))
    print('\tvalidation: %6d samples' % len(samples_val))
    print('\ttesting:    %6d samples' % len(samples_test))

    params = OrderedDict(numpy.load(path_load))
    del params['costs']
    t_params = OrderedDict()
    init_t_params(params, t_params)

    print('Building word sampler...')
    f_enc = build_enc(t_params, n_dim_img, n_dim_enc, n_dim_dec)
    f_dec = build_dec(t_params, n_dim_txt, n_dim_enc, n_dim_dec, n_dim_vocab, beam_size)

    print('Predicting...')
    preds_train, err_train = predict(f_enc, f_dec, samples_train, get_batches(len(samples_train), batch_size), mat_train, beam_size, max_len, 'PREDICT TRA')
    with open(path_out_train, 'w') as file_out:
        json.dump(preds_train, file_out)

    preds_val, err_val = predict(f_enc, f_dec, samples_val, get_batches(len(samples_val), batch_size), mat_val, beam_size, max_len, 'PREDICT VAL')
    with open(path_out_val, 'w') as file_out:
        json.dump(preds_val, file_out)

    preds_test, err_test = predict(f_enc, f_dec, samples_test, get_batches(len(samples_test), batch_size), mat_test, beam_size, max_len, 'PREDICT TES')
    with open(path_out_test, 'w') as file_out:
        json.dump(preds_test, file_out)

    print('ERR TRA: %f    ERR VAL: %f    ERR TES: %f' % (err_train, err_val, err_test))
    print('Done.')


if __name__ == '__main__':
    # For debugging, use the following arguments:
    main(max_samples_train=400, max_samples_val=50, max_samples_test=50, batch_size=10)
    # main()
