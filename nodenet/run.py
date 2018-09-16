# -*- coding: utf-8 -*- 

import numpy as np
import random
import time

from config import *
from optimize import adaGrad
from network import convNet

def run(train_data, test_data, params):

	new_params = train(train_data, params)
	test(test_data, new_params)

# TRAIN the network (50k MNIST images)
def train(train_data, params):

	cost = []
	acc = []

	start_time = time.time()

	for epoch in range(0, NUM_EPOCHS):

		np.random.shuffle(train_data)	

		batches = [train_data[k : k + BATCH_SIZE] for k in xrange(0, len(train_data), BATCH_SIZE)]
		
		i = 0
		for batch in batches:

			X, y = _unpack_data(batch)
		
			# train with AdaGrad optimizer
			new_params = adaGrad(X, y, params, cost, acc)
	
			epoch_acc = round(np.sum(acc[epoch * len(train_data) / BATCH_SIZE:]) / (i + 1), 2)
			
			print("Batch " + str(i + 1) + "/" + str(len(train_data) / BATCH_SIZE) +
				" of Epoch " + str(epoch + 1) + "/" + str(NUM_EPOCHS) +
				": Cost: %.2f" % cost[-1][0] + ", Batch: %.0f" % (acc[-1]*100) +
				"%% accuracy, Epoch: %.0f" % (epoch_acc * 100) + "% accuracy")
		
			i += 1

	end_time = time.time()
	print("Time to train: " + (end_time - start_time))

	return new_params

# TEST the network (10k MNIST images)
def test(test_data, params):

	X, y = _unpack_data(test_data)

	num_correct = 0

	start_time = time.time()

	for i in range(0, len(test_data)):

		image = X[i]
		
		label = np.zeros((NUM_OUTPUT_NODES, 1))
		label[int(y[i]), 0] = 1
		
		predict = convNet(image, label, params, runtype="test")	
		
		if predict == y[i]:
			num_correct += 1
			mark = "✓"
		else:
			mark = "⌧"

		print("#" + str(i + 1) + ": " + str(predict) + " | %.0f " % y[i] + mark)

	test_acc = float(num_correct) / len(test_data) * 100
	print("Testing accuracy: " + str(test_acc))

	end_time = time.time()
	print("Time to test: " + (end_time - start_time))

# HELPERS

# separate images and labels
def _unpack_data(data):
	X = data[:, 0:-1]
	X = X.reshape(len(data), IMAGE_DEPTH, IMAGE_WIDTH, IMAGE_WIDTH)
	y = data[:, -1]

	return [X, y]
