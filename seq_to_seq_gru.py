﻿import time
import numpy
from collections import OrderedDict
from theano import tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams


SEED = 65535


def numpy_floatX(data):
	return numpy.asarray(data, dtype=config.floatX)


def zipp(params, tparams):
	"""
	When we reload the model. Needed for the GPU stuff.
	"""
	for kk, vv in params.items():
		tparams[kk].set_value(vv)


def unzip(zipped):
	"""
	When we pickle the model. Needed for the GPU stuff.
	"""
	params = OrderedDict()
	for kk, vv in zipped.items():
		params[kk] = vv.get_value()

	return params


def concat(pp, name):
	return '%s_%s' % (pp, name)


def slice(x, idx, ndim):
	if x.ndim == 3:
		return x[:, :, idx * ndim: (idx + 1) * ndim]
	return x[:, idx * ndim: (idx + 1) * ndim]


def get_minibatches_idx(n, minibatch_size, shuffle=False):
	"""
	Used to shuffle the dataset at each iteration.
	"""
	idx_list = numpy.arange(n, dtype="int32")
	if shuffle:
		numpy.random.shuffle(idx_list)
	minibatches = []
	idx_start = 0
	while idx_start < len(idx_list):
		minibatches.append(idx_list[idx_start: idx_start + minibatch_size])
		idx_start += minibatch_size

	return zip(range(len(minibatches)), minibatches)


def init_tparams(params, tparams):
	for kk, pp in params.items():
		tparams[kk] = theano.shared(params[kk], name=kk)


def norm_weight(ndim_in, ndim_out, scale=0.01):
	return scale * numpy.random.randn(ndim_in, ndim_out).astype(config.floatX)


def layer_dropout(state_before, dropout, trng):
	return tensor.switch(dropout, (state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)), state_before * 0.5)


def layer_lstm(mask, state_below, tparams, ndim_in, ndim_out, prefix, init):
	"""
	LSTM layer
	"""
	def _step(m_, x_, h_, c_):
		preact = tensor.dot(h_, tparams[concat(prefix, 'U')]) + x_

		i = tensor.nnet.sigmoid(slice(preact, 0, ndim_out))
		f = tensor.nnet.sigmoid(slice(preact, 1, ndim_out))
		o = tensor.nnet.sigmoid(slice(preact, 2, ndim_out))
		c = tensor.tanh(slice(preact, 3, ndim_out))

		c = f * c_ + i * c
		c = m_[:, None] * c + (1. - m_)[:, None] * c_
		h = o * tensor.tanh(c)
		h = m_[:, None] * h + (1. - m_)[:, None] * h_

		return h, c

	if init:
		params = OrderedDict()
		params[concat(prefix, 'W')] = numpy.concatenate([norm_weight(ndim_in, ndim_out), norm_weight(ndim_in, ndim_out), norm_weight(ndim_in, ndim_out), norm_weight(ndim_in, ndim_out)], axis=1)
		params[concat(prefix, 'U')] = numpy.concatenate([norm_weight(ndim_out, ndim_out), norm_weight(ndim_out, ndim_out), norm_weight(ndim_out, ndim_out), norm_weight(ndim_out, ndim_out)], axis=1)
		params[concat(prefix, 'b')] = numpy.zeros((4 * ndim_out,)).astype(config.floatX)
		init_tparams(params, tparams)

	nsteps = state_below.shape[0]
	nsamples = state_below.shape[1] if state_below.ndim == 3 else 1
	state_below = (tensor.dot(state_below, tparams[concat(prefix, 'W')]) + tparams[concat(prefix, 'b')])
	rval, updates = theano.scan(_step, sequences=[mask, state_below], outputs_info=[tensor.alloc(numpy_floatX(0.), nsamples, ndim_out), tensor.alloc(numpy_floatX(0.), nsamples, ndim_out)], name=concat(prefix, '_layer'), n_steps=nsteps)

	return rval[0]


def layer_gru(mask, state_below, tparams, ndim_in, ndim_out, prefix, init):
	"""
	GRU layer
	"""
	def _step(m_, x_, h_):
		preact = tensor.dot(h_, tparams[concat(prefix, 'U')]) + x_

		r = tensor.nnet.sigmoid(slice(preact, 0, ndim_out))
		u = tensor.nnet.sigmoid(slice(preact, 1, ndim_out))
		h_new = tensor.tanh(tensor.dot(r * h_, slice(tparams[concat(prefix, 'U')], 2, ndim_out)) + slice(x_, 2, ndim_out))
		h = (1. - u) * h_ + u * h_new
		h = m_[:, None] * h + (1. - m_)[:, None] * h_

		return h

	if init:
		params = OrderedDict()
		params[concat(prefix, 'W')] = numpy.concatenate([norm_weight(ndim_in, ndim_out), norm_weight(ndim_in, ndim_out), norm_weight(ndim_in, ndim_out)], axis=1)
		params[concat(prefix, 'U')] = numpy.concatenate([norm_weight(ndim_out, ndim_out), norm_weight(ndim_out, ndim_out), norm_weight(ndim_out, ndim_out)], axis=1)
		params[concat(prefix, 'b')] = numpy.zeros((3 * ndim_out,)).astype(config.floatX)
		init_tparams(params, tparams)

	nsteps = state_below.shape[0]
	nsamples = state_below.shape[1] if state_below.ndim == 3 else 1
	state_below = (tensor.dot(state_below, tparams[concat(prefix, 'W')]) + tparams[concat(prefix, 'b')])
	rval, updates = theano.scan(_step, sequences=[mask, state_below], outputs_info=[tensor.alloc(numpy_floatX(0.), nsamples, ndim_out)], name=concat(prefix, '_layer'), n_steps=nsteps)

	return rval


def sgd(lr, tparams, grads, x, mask_x, y, mask_y, cost):
	"""
	Stochastic Gradient Descent

	Parameters
	----------
	lr: Theano SharedVariable
		Initial learning rate
	tparams: Theano SharedVariable
		Model parameters
	grads: Theano variable
		Gradients of cost w.r.t to parameres
	x: Theano variable
		Model inputs
	mask_x: Theano variable
		Sequence mask
	y: Theano variable
		Targets
	mask_y: Theano variable
		Sequence mask
	cost: Theano variable
		Objective fucntion to minimize

	NOTE: A more complicated version of sgd then needed. This is done like that for adadelta and rmsprop.
	"""
	# New set of shared variable that will contain the gradient for a mini-batch.
	gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in tparams.items()]
	gsup = [(gs, g) for gs, g in zip(gshared, grads)]

	# Function that computes gradients for a mini-batch, but do not updates the weights.
	f_grad_shared = theano.function([x, mask_x, y, mask_y], cost, updates=gsup, name='sgd_f_grad_shared')
	pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

	# Function that updates the weights from the previously computed gradient.
	f_update = theano.function([lr], [], updates=pup, name='sgd_f_update')

	return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask_x, y, mask_y, cost):
	"""
	An adaptive learning rate optimizer

	Parameters
	----------
	lr: Theano SharedVariable
		Initial learning rate
	tparams: Theano SharedVariable
		Model parameters
	grads: Theano variable
		Gradients of cost w.r.t. to parameres
	x: Theano variable
		Model inputs
	mask_x: Theano variable
		Sequence mask
	y: Theano variable
		Targets
	mask_y: Theano variable
		Sequence mask
	cost: Theano variable
		Objective fucntion to minimize

	NOTE: For more information, see Matthew D. Zeiler. ADADELTA: An Adaptive Learning Rate Method. arXiv:1212.5701.
	"""

	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.items()]
	running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in tparams.items()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tparams.items()]
	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	f_grad_shared = theano.function([x, mask_x, y, mask_y], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
	updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

	f_update = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

	return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask_x, y, mask_y, cost):
	"""
	A variant of SGD that scales the step size by running average of the recent step norms.

	Parameters
	----------
	lr: Theano SharedVariable
		Initial learning rate
	tparams: Theano SharedVariable
		Model parameters
	grads: Theano variable
		Gradients of cost w.r.t to parameres
	x: Theano variable
		Model inputs
	mask_x: Theano variable
		Sequence mask
	y: Theano variable
		Targets
	mask_y: Theano variable
		Sequence mask
	cost: Theano variable
		Objective fucntion to minimize

	NOTE: For more information, see Geoff Hinton. Neural Networks for Machine Learning, lecture 6a. http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf.
	"""

	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in tparams.items()]
	running_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad' % k) for k, p in tparams.items()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in tparams.items()]
	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	f_grad_shared = theano.function([x, mask_x, y, mask_y], cost, updates=zgup + rgup + rg2up, name='rmsprop_f_grad_shared')
	updir = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_updir' % k) for k, p in tparams.items()]
	updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
	param_up = [(p, p + udn[1]) for p, udn in zip(tparams.values(), updir_new)]

	f_update = theano.function([lr], [], updates=updir_new + param_up, on_unused_input='ignore', name='rmsprop_f_update')

	return f_grad_shared, f_update


def build_model(tparams, ndim_x, ndim_y, ndim_enc, ndim_dec, path_load):
	"""
	Build the whole model
	"""
	trng = MRG_RandomStreams(SEED)
	# Used for dropout
	dropout = theano.shared(numpy_floatX(0.))
	x = tensor.tensor3('x', dtype=config.floatX)
	mask_x = tensor.matrix('mask_x', dtype='int8')
	y = tensor.tensor3('y', dtype='int8')
	mask_y = tensor.matrix('mask_y', dtype='int8')

	tparams = OrderedDict()
	if path_load:
		init_tparams(numpy.load(path_load), tparams)
	else:
		tparams['fc_U'] = theano.shared(norm_weight(ndim_dec, ndim_y))
		tparams['fc_b'] = theano.shared(numpy.zeros((ndim_y,)).astype(config.floatX))

	encoder_1 = layer_gru(mask_x, x, tparams, ndim_x, ndim_enc, prefix='encoder_1', path_load is None)[-1]
	encoder_1 = tensor.repeat(encoder_1.dimshuffle('x', 0, 1), y.shape[0], axis=0)
	decoder_1 = layer_gru(mask_y, encoder_1, tparams, ndim_enc, ndim_dec, prefix='decoder_1', path_load is None)
	fc = tensor.dot(proj, tparams['fc_U']) + tparams['fc_b']
	
	pred = tensor.exp(fc - tensor.max(fc, axis=-1, keepdims=True))
	pred /= tensor.sum(pred, axis=-1, keepdims=True)

	f_pred_prob = theano.function([x, mask_x, mask_y], pred, name='f_pred_prob')
	f_pred = theano.function([x, mask_x, mask_y], pred.argmax(axis=1), name='f_pred')

	off = 1e-8
	if pred.dtype == 'float16':
		off = 1e-6

	cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

	return dropout, x, mask_x, y, mask_y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
	""" If you want to use a trained model, this is useful to compute
	the probabilities of new examples.
	"""
	n_samples = len(data[0])
	probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

	n_done = 0

	for _, valid_index in iterator:
		x, mask, y = prepare_data([data[0][t] for t in valid_index],
								  numpy.array(data[1])[valid_index],
								  maxlen=None)
		pred_probs = f_pred_prob(x, mask)
		probs[valid_index, :] = pred_probs

		n_done += len(valid_index)
		if verbose:
			print('%d/%d samples classified' % (n_done, n_samples))

	return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
	"""
	Just compute the error
	f_pred: Theano fct computing the prediction
	prepare_data: usual prepare_data for that dataset.
	"""
	valid_err = 0
	for _, valid_index in iterator:
		x, mask, y = prepare_data([data[0][t] for t in valid_index],
								  numpy.array(data[1])[valid_index],
								  maxlen=None)
		preds = f_pred(x, mask)
		targets = numpy.array(data[1])[valid_index]
		valid_err += (preds == targets).sum()
	valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

	return valid_err


def train_lstm(
	dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
	patience=10,  # Number of epoch to wait before early stop if no progress
	max_epochs=5000,  # The maximum number of epoch to run
	dispFreq=10,  # Display to stdout the training progress every N updates
	decay_c=0.,  # Weight decay for the classifier applied to the U weights.
	lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
	n_words=10000,  # Vocabulary size
	optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
	encoder='lstm',  # TODO: can be removed must be lstm.
	saveto='lstm_model.npz',  # The best model will be saved there
	validFreq=370,  # Compute the validation error after this number of update.
	saveFreq=1110,  # Save the parameters after every saveFreq updates
	maxlen=100,  # Sequence longer then this get ignored
	batch_size=16,  # The batch size during training.
	valid_batch_size=64,  # The batch size used for validation/test set.
	dataset='imdb',

	# Parameter for extra option
	noise_std=0.,
	use_dropout=True,  # if False slightly faster, but worst test error
					   # This frequently need a bigger model.
	reload_model=None,  # Path to a saved model we want to start from.
	test_size=-1,  # If >0, we keep only this number of test example.
):

	# Model options
	model_options = locals().copy()
	print("model options", model_options)

	print('Loading data')
	train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
								   maxlen=maxlen)
	if test_size > 0:
		# The test set is sorted by size, but we want to keep random
		# size example.  So we must select a random selection of the
		# examples.
		idx = numpy.arange(len(test[0]))
		numpy.random.shuffle(idx)
		idx = idx[:test_size]
		test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

	ydim = numpy.max(train[1]) + 1

	model_options['ydim'] = ydim

	print('Building model')
	# This create the initial parameters as numpy ndarrays.
	# Dict name (string) -> numpy ndarray
	# This create Theano Shared Variable from the parameters.
	# Dict name (string) -> Theano Tensor Shared Variable
	# params and tparams have different copy of the weights.
	tparams = init_tparams(params)

	# dropout is for dropout
	(dropout, x, mask,
	 y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

	if decay_c > 0.:
		decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
		weight_decay = 0.
		weight_decay += (tparams['U'] ** 2).sum()
		weight_decay *= decay_c
		cost += weight_decay

	f_cost = theano.function([x, mask, y], cost, name='f_cost')

	grads = tensor.grad(cost, wrt=list(tparams.values()))
	f_grad = theano.function([x, mask, y], grads, name='f_grad')

	lr = tensor.scalar(name='lr')
	f_grad_shared, f_update = optimizer(lr, tparams, grads,
										x, mask, y, cost)

	print('Optimization')

	kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
	kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

	print("%d train examples" % len(train[0]))
	print("%d valid examples" % len(valid[0]))
	print("%d test examples" % len(test[0]))

	history_errs = []
	best_p = None
	bad_count = 0

	if validFreq == -1:
		validFreq = len(train[0]) // batch_size
	if saveFreq == -1:
		saveFreq = len(train[0]) // batch_size

	uidx = 0  # the number of update done
	estop = False  # early stop
	start_time = time.time()
	try:
		for eidx in range(max_epochs):
			n_samples = 0

			# Get new shuffled index for the training set.
			kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

			for _, train_index in kf:
				uidx += 1
				dropout.set_value(1.)

				# Select the random examples for this minibatch
				y = [train[1][t] for t in train_index]
				x = [train[0][t]for t in train_index]

				# Get the data in numpy.ndarray format
				# This swap the axis!
				# Return something of shape (minibatch maxlen, n samples)
				x, mask, y = prepare_data(x, y)
				n_samples += x.shape[1]

				cost = f_grad_shared(x, mask, y)
				f_update(lrate)

				if numpy.isnan(cost) or numpy.isinf(cost):
					print('bad cost detected: ', cost)
					return 1., 1., 1.

				if numpy.mod(uidx, dispFreq) == 0:
					print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

				if saveto and numpy.mod(uidx, saveFreq) == 0:
					print('Saving...')

					if best_p is not None:
						params = best_p
					else:
						params = unzip(tparams)
					numpy.savez(saveto, history_errs=history_errs, **params)
					pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
					print('Done')

				if numpy.mod(uidx, validFreq) == 0:
					dropout.set_value(0.)
					train_err = pred_error(f_pred, prepare_data, train, kf)
					valid_err = pred_error(f_pred, prepare_data, valid,
										   kf_valid)
					test_err = pred_error(f_pred, prepare_data, test, kf_test)

					history_errs.append([valid_err, test_err])

					if (best_p is None or
						valid_err <= numpy.array(history_errs)[:,
															   0].min()):

						best_p = unzip(tparams)
						bad_counter = 0

					print( ('Train ', train_err, 'Valid ', valid_err,
						   'Test ', test_err) )

					if (len(history_errs) > patience and
						valid_err >= numpy.array(history_errs)[:-patience,
															   0].min()):
						bad_counter += 1
						if bad_counter > patience:
							print('Early Stop!')
							estop = True
							break

			print('Seen %d samples' % n_samples)

			if estop:
				break

	except KeyboardInterrupt:
		print("Training interupted")

	end_time = time.time()
	if best_p is not None:
		zipp(best_p, tparams)
	else:
		best_p = unzip(tparams)

	dropout.set_value(0.)
	kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
	train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
	valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
	test_err = pred_error(f_pred, prepare_data, test, kf_test)

	print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
	if saveto:
		numpy.savez(saveto, train_err=train_err,
					valid_err=valid_err, test_err=test_err,
					history_errs=history_errs, **best_p)
	print('The code run for %d epochs, with %f sec/epochs' % (
		(eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
	print(('Training took %.1fs' % (end_time - start_time)))
	return train_err, valid_err, test_err


if __name__ == '__main__':
	# See function train for all possible parameter and there definition.
	numpy.random.seed(SEED)
	train_lstm(
		max_epochs=100,
		test_size=500,
	)
