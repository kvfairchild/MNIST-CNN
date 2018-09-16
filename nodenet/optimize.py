import numpy as np

from config import *
from network import convNet

def adaGrad(X, y, params, cost, acc):

	[filt1, filt2, bias1, bias2, theta3, bias3] = params

	# init grad params
	n_correct = 0
	cost_ = 0

	dtheta3 = np.zeros(theta3.shape)
	dbias3 = np.zeros(bias3.shape)

	for i in range(0, BATCH_SIZE):
		
		image = X[i]

		label = np.zeros((theta3.shape[0],1))
		label[int(y[i]), 0] = 1
		
		# Get gradients for all current parameters
		[dfilt1_, dfilt2_, dbias1_, dbias2_, dtheta3_, dbias3_, curr_cost, acc_] = convNet(image, label, params, runtype="train")

		cost_ += curr_cost
		n_correct += acc_

		_update(theta3, dtheta3, dtheta3_)
		_update(bias3, dbias3, dbias3_)

	cost_ = cost_ / BATCH_SIZE
	cost.append(cost_)
	accuracy = float(n_correct) / BATCH_SIZE
	acc.append(accuracy)

	return params

def _update(param, dparam, dparam_):
	eps = np.finfo(float).eps

	dparam += dparam_ ** 2 
	param -= LEARNING_RATE * dparam_ / np.sqrt(dparam + eps) / BATCH_SIZE

