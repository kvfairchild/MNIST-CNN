import numpy as np

def convNet(image, label, params, runtype):

	# Returns gradient for all parameters in each training iteration

	[filt1, filt2, bias1, bias2, theta3, bias3] = params

	## l - channel
	## w - size of square image
	## l1 - # filters in Conv1
	## l2 - # filters in Conv2
	## w1 - size of image after conv1
	## w2 - size of image after conv2

	(l, w, w) = image.shape		
	l1 = len(filt1)
	l2 = len(filt2)
	(_, f, f) = filt1[0].shape
	w1 = w - f + 1
	w2 = w1 - f + 1
	
	# First convolution layer
	conv1 = _convolution(image, l1, w1, f, filt1, bias1)

	# Pooled layer with 2 * 2 size and 2, 2 stride
	pooled_layer = _maxpool(conv1, 2, 2)		

	# Second convolution layer
	conv2 = _convolution(image, l2, w2, f, filt2, bias2)

	# Pooled layer with 2 * 2 size and 2, 2 stride
	pooled_layer = _maxpool(conv2, 2, 2)	

	# Fully connected layer
	fc1 = pooled_layer.reshape(((w2/2) * (w2/2) * l2, 1))
	
	output = theta3.dot(fc1) + bias3
	
	if runtype == "train":

		# softmax to get cost
		cost, probs = _softmax(output, label, runtype)

		if np.argmax(output) == np.argmax(label):
			acc = 1
		else:
			acc = 0

		dout = probs - label # output deriv for backprop

		return _backpropagate(dout, fc1, bias3, theta3, l2, w2, conv2, conv1, l1, w1, f, l, filt2, image, cost, acc)
	
	elif runtype == "test":

		# softmax to get prediction
		return _softmax(output, label, runtype)

	else:
		raise ValueError ("runtype must be either 'train' or 'test'")


def _backpropagate(dout, fc1, bias3, theta3, l2, w2, conv2, conv1, l1, w1, f, l, filt2, image, cost, acc):

	dtheta3 = dout.dot(fc1.T) # theta deriv = dout * transpose of fc1 array
	dbias3 = sum(dout.T).T.reshape((10, 1))

	dfc1 = theta3.T.dot(dout)
	dpool = dfc1.T.reshape((l2, w2/2, w2/2))

	# backprop conv2
	dconv2 = np.zeros((l2, w2, w2))
	_backprop_conv_out(l2, w2, conv2, dconv2, dpool)

	# backprop conv1
	dconv1 = np.zeros((l1, w1, w1))
	[dfilt2, dbias2] = _backprop_conv(l2, l1, f, w2, dconv2, dconv1, conv1, filt2, 0)
	[dfilt1, dbias1] = _backprop_conv(l1, l, f, w1, dconv1, image, conv1, filt2, 1)

	return [dfilt1, dfilt2, dbias1, dbias2, dtheta3, dbias3, cost, acc]


# LAYERS

def _convolution(image, l, w, f, filt, bias):
	conv = np.zeros((l, w, w))

	for jj in range(0, l):
		for x in range(0, w):
			for y in range(0, w):
				conv[jj, x, y] = np.sum(image[:, x:x+f, y:y+f] * filt[jj]) + bias[jj]

	conv[conv <= 0] = 0 # ReLu

	return conv

def _maxpool(X, f, s):
	(l, w, w) = X.shape
	pool = np.zeros((l, (w-f)/s+1, (w-f)/s+1))
	for jj in range(0, l):
		i = 0
		while (i < w):
			j = 0
			while(j < w):
				pool[jj,i/2,j/2] = np.max(X[jj, i:i+f, j:j+f])
				j += s
			i += s

	return pool


# HELPERS

# if training, return cost & probs
# if testing, return prediction
def _softmax(out, y, runtype):
	eout = np.exp(out, dtype=np.float)
	probs = eout/sum(eout)
	
	if runtype == "train":
		p = sum(y * probs)
		cost = -np.log(p)

		return cost, probs

	elif runtype == "test":
		return np.argmax(probs)

# BACKPROP HELPERS

# backprop last layer wrt max index output
def _backprop_conv_out(l2, w2, conv2, dconv2, dpool):

	for jj in range(0, l2):
		i = 0
		while(i < w2):
			j = 0
			while(j < w2):
				idx = np.argmax(conv2[jj, i:i+2, j:j+2], axis=None)
				(a, b) = np.unravel_index(idx, conv2[jj, i:i+2, j:j+2].shape)
				dconv2[jj, i+a, j+b] = dpool[jj, i/2, j/2]
				j += 2
			i += 2
	
	dconv2[conv2 <= 0] = 0 # ReLu

# backprop earlier conv layers
def _backprop_conv(num_filt, prev_num_filt, f, w, dconv_out, dconv_in, conv_in, filt, i):

	# init filter & bias params
	dfilt = {}
	dbias = {}
	for xx in range(0, num_filt):
		dfilt[xx] = np.zeros((prev_num_filt, f, f))
		dbias[xx] = 0

	# backprop
	for jj in range(0, num_filt):
		for x in range(0, w):
			for y in range(0, w):
				dfilt[jj] += dconv_out[jj, x, y] * dconv_in[:, x:x+f, y:y+f]
				if (i == 0):
					dconv_in[:, x:x+f, y:y+f] += dconv_out[jj, x, y] * filt[jj]
	
		dbias[jj] = np.sum(dconv_out[jj])
	
	if (i == 0):
		dconv_in[conv_in <= 0] = 0 # ReLu

	return [dfilt, dbias]
