from __future__ import print_function
import time
import json
import numpy
import theano
from collections import OrderedDict
from theano import tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams


SEED = 65535


def get_batches(n_sample, batch_size, shuffle=False):
	"""
	Get batches of data
	"""
	indices = numpy.arange(n_sample, dtype="int32")
	if shuffle:
		numpy.random.shuffle(indices)
	batches = []
	idx_start = 0
	while idx_start < n_sample:
		batches.append(indices[idx_start: idx_start + batch_size])
		idx_start += batch_size

	return batches


def load_data(path_list, path_mat, n_dim_y):
	"""
	Load dataset
	"""
	with open(path_list, 'r') as input:
		samples = json.load(input)
		for i in range(len(samples)):
			samples[i][0] = samples[i][0][: : -1]
			n_dim_y = max(n_dim_y, max(samples[i][1]) + 1)

	return samples, numpy.load(path_mat), n_dim_y


def load_batch(samples, batch, mat):
	"""
	get and expand data of a batch in NumPy ndarray format
	"""
	x = []
	y = []
	for idx in batch:
		x.append(samples[idx][0])
		y.append(samples[idx][1])
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

	return numpy.swapaxes(numpy.asarray(x, dtype=config.floatX), 0, 1), numpy.asarray(mask_x, dtype='int8').T, numpy.asarray(y, dtype='int32').T, numpy.asarray(mask_y, dtype='int8').T


def numpy_floatX(data):
	return numpy.asarray(data, dtype=config.floatX)


def zipp(params, t_params):
	"""
	When we reload the model. Needed for the GPU stuff.
	"""
	for key, value in params.items():
		t_params[key].set_value(value)


def unzip(zipped):
	"""
	When we pickle the model. Needed for the GPU stuff.
	"""
	params = OrderedDict()
	for key, value in zipped.items():
		params[key] = value.get_value()

	return params


def concat(prefix, name):
	return '%s_%s' % (prefix, name)


def slice(x, idx, n_dim):
	if x.ndim == 3:
		return x[:, :, idx * n_dim: (idx + 1) * n_dim]
	return x[:, idx * n_dim: (idx + 1) * n_dim]


def init_t_params(params, t_params):
	for key, value in params.items():
		t_params[key] = theano.shared(value, name=key)


def norm_weight(n_dim_in, n_dim_out, scale=0.01):
	return scale * numpy.random.randn(n_dim_in, n_dim_out).astype(config.floatX)


def layer_dropout(state_before, use_dropout, trng):
	return tensor.switch(use_dropout, (state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)), state_before * 0.5)


def layer_fc(state_in, t_params, n_dim_in, n_dim_out, prefix, init):
	"""
	Full-connected layer
	"""
	if init:
		params = OrderedDict()
		params[concat(prefix, 'W')] = norm_weight(n_dim_in, n_dim_out)
		params[concat(prefix, 'b')] = numpy.zeros((n_dim_out,)).astype(config.floatX)
		init_t_params(params, t_params)

	return tensor.dot(state_in, t_params[concat(prefix, 'W')]) + t_params[concat(prefix, 'b')]


def layer_softmax(state_in):
	"""
	Softmax layer
	"""
	state_out = tensor.exp(state_in - state_in.max(axis=-1, keepdims=True))

	return state_out / state_out.sum(axis=-1, keepdims=True)


def layer_lstm(mask, state_below, t_params, n_dim_in, n_dim_out, prefix, init):
	"""
	LSTM layer
	"""
	def _step(m_, x_, h_, c_):
		preact = tensor.dot(h_, t_params[concat(prefix, 'U')]) + x_

		i = tensor.nnet.sigmoid(slice(preact, 0, n_dim_out))
		f = tensor.nnet.sigmoid(slice(preact, 1, n_dim_out))
		o = tensor.nnet.sigmoid(slice(preact, 2, n_dim_out))
		c = tensor.tanh(slice(preact, 3, n_dim_out))

		c = f * c_ + i * c
		c = m_[:, None] * c + (1. - m_)[:, None] * c_
		h = o * tensor.tanh(c)
		h = m_[:, None] * h + (1. - m_)[:, None] * h_

		return h, c

	if init:
		params = OrderedDict()
		params[concat(prefix, 'W')] = numpy.concatenate([norm_weight(n_dim_in, n_dim_out), norm_weight(n_dim_in, n_dim_out), norm_weight(n_dim_in, n_dim_out), norm_weight(n_dim_in, n_dim_out)], axis=1)
		params[concat(prefix, 'U')] = numpy.concatenate([norm_weight(n_dim_out, n_dim_out), norm_weight(n_dim_out, n_dim_out), norm_weight(n_dim_out, n_dim_out), norm_weight(n_dim_out, n_dim_out)], axis=1)
		params[concat(prefix, 'b')] = numpy.zeros((4 * n_dim_out,)).astype(config.floatX)
		init_t_params(params, t_params)

	n_steps = state_below.shape[0]
	n_samples = state_below.shape[1] if state_below.ndim == 3 else 1
	state_below = (tensor.dot(state_below, t_params[concat(prefix, 'W')]) + t_params[concat(prefix, 'b')])
	rval, updates = theano.scan(_step, sequences=[mask, state_below], outputs_info=[tensor.alloc(numpy_floatX(0.), n_samples, n_dim_out), tensor.alloc(numpy_floatX(0.), n_samples, n_dim_out)], name=concat(prefix, '_layer'), n_steps=n_steps)

	return rval[0]


def layer_gru(mask, state_below, t_params, n_dim_in, n_dim_out, prefix, init):
	"""
	GRU layer
	"""
	def _step(m_, x_, h_):
		preact = tensor.dot(h_, t_params[concat(prefix, 'U')]) + x_

		r = tensor.nnet.sigmoid(slice(preact, 0, n_dim_out))
		u = tensor.nnet.sigmoid(slice(preact, 1, n_dim_out))
		h_new = tensor.tanh(tensor.dot(r * h_, slice(t_params[concat(prefix, 'U')], 2, n_dim_out)) + slice(x_, 2, n_dim_out))
		h = (1. - u) * h_ + u * h_new
		h = m_[:, None] * h + (1. - m_)[:, None] * h_

		return h

	if init:
		params = OrderedDict()
		params[concat(prefix, 'W')] = numpy.concatenate([norm_weight(n_dim_in, n_dim_out), norm_weight(n_dim_in, n_dim_out), norm_weight(n_dim_in, n_dim_out)], axis=1)
		params[concat(prefix, 'U')] = numpy.concatenate([norm_weight(n_dim_out, n_dim_out), norm_weight(n_dim_out, n_dim_out), norm_weight(n_dim_out, n_dim_out)], axis=1)
		params[concat(prefix, 'b')] = numpy.zeros((3 * n_dim_out,)).astype(config.floatX)
		init_t_params(params, t_params)

	n_steps = state_below.shape[0]
	n_samples = state_below.shape[1] if state_below.ndim == 3 else 1
	state_below = (tensor.dot(state_below, t_params[concat(prefix, 'W')]) + t_params[concat(prefix, 'b')])
	rval, updates = theano.scan(_step, sequences=[mask, state_below], outputs_info=[tensor.alloc(numpy_floatX(0.), n_samples, n_dim_out)], name=concat(prefix, '_layer'), n_steps=n_steps)

	return rval


def sgd(lr, t_params, grads, x, mask_x, y, mask_y, cost):
	"""
	Stochastic Gradient Descent

	Parameters
	----------
	lr: Theano SharedVariable
		Initial learning rate
	t_params: Theano SharedVariable
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
	gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k) for k, p in t_params.items()]
	gsup = [(gs, g) for gs, g in zip(gshared, grads)]

	# Function that computes gradients for a mini-batch, but do not updates the weights.
	f_grad_shared = theano.function([x, mask_x, y, mask_y], cost, updates=gsup, name='sgd_f_grad_shared')
	pup = [(p, p - lr * g) for p, g in zip(t_params.values(), gshared)]

	# Function that updates the weights from the previously computed gradient.
	f_update = theano.function([lr], [], updates=pup, name='sgd_f_update')

	return f_grad_shared, f_update


def adadelta(lr, t_params, grads, x, mask_x, y, mask_y, cost):
	"""
	An adaptive learning rate optimizer

	Parameters
	----------
	lr: Theano SharedVariable
		Initial learning rate
	t_params: Theano SharedVariable
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

	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in t_params.items()]
	running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rup2' % k) for k, p in t_params.items()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in t_params.items()]
	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	f_grad_shared = theano.function([x, mask_x, y, mask_y], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')
	updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
	ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
	param_up = [(p, p + ud) for p, ud in zip(t_params.values(), updir)]

	f_update = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')

	return f_grad_shared, f_update


def rmsprop(lr, t_params, grads, x, mask_x, y, mask_y, cost):
	"""
	A variant of SGD that scales the step size by running average of the recent step norms.

	Parameters
	----------
	lr: Theano SharedVariable
		Initial learning rate
	t_params: Theano SharedVariable
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

	zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_grad' % k) for k, p in t_params.items()]
	running_grads = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad' % k) for k, p in t_params.items()]
	running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_rgrad2' % k) for k, p in t_params.items()]
	zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
	rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
	rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

	f_grad_shared = theano.function([x, mask_x, y, mask_y], cost, updates=zgup + rgup + rg2up, name='rmsprop_f_grad_shared')
	updir = [theano.shared(p.get_value() * numpy_floatX(0.), name='%s_updir' % k) for k, p in t_params.items()]
	updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
	param_up = [(p, p + udn[1]) for p, udn in zip(t_params.values(), updir_new)]

	f_update = theano.function([lr], [], updates=updir_new + param_up, on_unused_input='ignore', name='rmsprop_f_update')

	return f_grad_shared, f_update


def pred_error(f_prob, samples, batches, mat):
	"""
	Compute the prediction error
	"""
	errs = []
	for batch in batches:
		x, mask_x, y, mask_y = load_batch(samples, batch, mat)
		preds = f_prob(x, mask_x, y, mask_y).argmax(axis=-1)
		mat_err = (preds != y) * mask_y
		for i in range(mat_err.shape[0]):
			errs.apppend(mat_err[i].sum() / mask_y[i].sum())

	return preds, numpy.mean(errs)


def main_model(
	# Dataset Configuration
	path_train='train.json',			# Path to load training set
	path_val='val.json',				# Path to load validation set
	path_test='test.json',				# Path to load testing set
	path_mat_train='VGG19_train.npy',	# Path of image features of training set
	path_mat_val='VGG19_val.npy',		# Path of image features of validation set
	path_mat_test='VGG19_test.npy',		# Path of image features of testing set
	path_out_train='out_train.npy',		# Path to save predicted sentences of training set
	path_out_val='out_val.npy',			# Path to save predicted sentences of validation set
	path_out_test='out_test.npy',		# Path to save predicted sentences of testing set
	# Model configuration
	n_dim_img=4096,						# Image feature dimension
	n_dim_txt=-1,						# Vocabulary size
	n_dim_enc=256,						# Number of hidden units in encoder
	n_dim_dec=256,						# Number of hidden units in decoder
	batch_size_train=64,				# batch size in training
	batch_size_val=256,					# batch size in validation
	batch_size_test=256,				# batch size in testing
	optimizer=adadelta,					# [sgd|adadelta|rmsprop], sgd not recommanded
	lrate=0.0001,						# Learning rate for optimizer
	max_epochs=5000,					# Maximum number of epoch to run
	patience=10,						# Number of epoch to wait before early stop if no progress
	# MISC
	freq_disp=10,						# Display the training progress after this number of updates
	freq_valid=370,						# Compute the validation error after this number of updates
	freq_save=1110,						# Save the parameters after this number of updates
	path_load=None,						# Path to load a previouly trained model
	path_save='model.npz',				# Path to save the best model
):
	print('Loading data...')
	samples_train, mat_train, n_dim_txt = load_data(path_train, path_mat_train, n_dim_txt)
	samples_val, mat_val, n_dim_txt = load_data(path_val, path_mat_val, n_dim_txt)
	samples_test, mat_test, n_dim_txt = load_data(path_test, path_mat_test, n_dim_txt)

	print('Building model...')
	numpy.random.seed(SEED)
	trng = MRG_RandomStreams(SEED)
	# Flag of dropout
	use_dropout = theano.shared(numpy_floatX(0.))
	# Inputs
	x = tensor.tensor3('x', dtype=config.floatX)
	mask_x = tensor.matrix('mask_x', dtype='int8')
	y = tensor.matrix('y', dtype='int32')
	mask_y = tensor.matrix('mask_y', dtype='int8')

	t_params = OrderedDict()
	if path_load:
		init_t_params(numpy.load(path_load), t_params)
	# Encoder(s)
	output = layer_gru(mask_x, x, t_params, n_dim_img, n_dim_enc, 'encoder_1', path_load is None)[-1]
	# Repetition of the final state of hidden layer
	output = tensor.repeat(output.dimshuffle('x', 0, 1), y.shape[0], axis=0)
	# Decoder(s)
	output = layer_gru(mask_y, output, t_params, n_dim_enc, n_dim_dec, 'decoder_1', path_load is None)
	output = layer_dropout(output, use_dropout, trng)
	# Classifier
	output = layer_fc(output, t_params, n_dim_dec, n_dim_txt, 'fc', path_load is None)
	output = layer_softmax(output)
	# Cost function
	offset = 1e-8 if output.dtype == 'float16' else 1e-6
	cost = output.reshape((y.shape[0] * y.shape[1], n_dim_txt))[tensor.arange(y.shape[0] * y.shape[1]), y.flatten()].reshape((y.shape[0], y.shape[1]))
	cost = ((-tensor.log(cost + offset) * mask_y).sum(axis=0) / mask_y.sum(axis=0)).mean()
	grads = tensor.grad(cost, wrt=list(t_params.values()))
	f_prob = theano.function([x, mask_x, y, mask_y], output, name='f_prob')
	f_grad_shared, f_update = optimizer(tensor.scalar(name='lr'), t_params, grads, x, mask_x, y, mask_y, cost)

	print('Optimization...')
	print('%d training samples' % len(samples_train))
	print('%d validation samples' % len(samples_val))
	print('%d testing samples' % len(samples_test))
	batches_valid = get_batches(len(samples_val), batch_size_val)
	batches_test = get_batches(len(samples_test), batch_size_test)
	history_errs = []
	best_p = None
	bad_count = 0
	if freq_valid <= 0:
		freq_valid = len(samples_train) / batch_size_train
	if freq_save <= 0:
		freq_save = len(samples_train) / batch_size_train
	n_updates = 0	# the number of update done
	stop = False	# early stop
	time_start = time.time()

	for epoch in range(max_epochs):
		n_samples = 0
		# Get new shuffled index for the training set.
		batches_train = get_batches(len(samples_train), batch_size_train, shuffle=True)
		for batch in batches_train:
			n_updates += 1
			use_dropout.set_value(1.)
			x, mask_x, y, mask_y = load_batch(samples_train, batch, mat_train)
			n_samples += x.shape[1]
			cost = f_grad_shared(x, mask_x, y, mask_y)
			f_update(lrate)
			
			if numpy.isnan(cost) or numpy.isinf(cost):
				print('Bad cost detected: ', cost)
				return

			if numpy.mod(n_updates, freq_disp) == 0:
				print('Epoch: ', epoch, 'Update: ', n_updates, 'Cost: ', cost)

			if path_save and numpy.mod(n_updates, freq_save) == 0:
				print('Saving...')
				if best_p is not None:
					params = best_p
				else:
					params = unzip(t_params)
				numpy.save(path_save, params)
				print('Done.')

			if numpy.mod(n_updates, freq_valid) == 0:
				use_dropout.set_value(0.)
				err_train = pred_error(f_prob, samples_train, batches_train, mat_train)
				err_val = pred_error(f_prob, samples_val, batches_val, mat_val)
				err_test = pred_error(f_prob, samples_test, batches_test, mat_test)
				history_errs.append([err_val, err_test])

				if (best_p is None or err_val <= numpy.array(history_errs)[:, 0].min()):
					best_p = unzip(t_params)
					bad_counter = 0
				print('Training: ', err_train, 'Validation: ', err_val, 'Testing: ', err_test)
				if (len(history_errs) > patience and err_val >= numpy.array(history_errs)[:-patience, 0].min()):
					bad_counter += 1
					if bad_counter > patience:
						print('Early stop!')
						stop = True
						break
		if stop:
			break

	time_end = time.time()
	if best_p is not None:
		zipp(best_p, t_params)
	else:
		best_p = unzip(t_params)

	use_dropout.set_value(0.)
	batches_train = get_batches(len(samples_train), batch_size_train)
	preds_train, err_train = pred_error(f_prob, samples_train, batches_train, mat_train)
	preds_val, err_val = pred_error(f_prob, samples_val, batches_val, mat_val)
	preds_test, err_test = pred_error(f_prob, samples_test, batches_test, mat_test)

	print('Training: ', err_train, 'Validation: ', err_val, 'Testing: ', err_test)
	if path_save:
		numpy.save(path_save, best_p)
	numpy.save(path_out_train, preds_train)
	numpy.save(path_out_val, preds_val)
	numpy.save(path_out_test, preds_test)
	print('The code run for %d epochs, with %f sec/epochs' % ((epoch + 1), (time_end - time_start) / (1. * (epoch + 1))))
	print(('Training took %.1fs' % (time_end - time_start)))
	return err_train, err_val, err_test


if __name__ == '__main__':
	main_model()