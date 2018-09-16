#!/usr/bin/env python

from nodenet.config import *
from nodenet.initialize import *
from nodenet.run import *
import MNIST_file_parser as fp

if __name__ == "__main__":

	train_data = fp.read("training")
	test_data = fp.read("testing")

	params = initialize_net()

	run(train_data, test_data, params)
