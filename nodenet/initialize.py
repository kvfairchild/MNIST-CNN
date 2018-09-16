import numpy as np
import scipy.stats

from config import *

def initialize_net():

    print "Initializing CNN..."

    # init parameters
    filt1 = {}
    filt2 = {}
    bias1 = {}
    bias2 = {}

    for i in range(0, NUM_FILT1):
        filt1[i] = _initialize_truncated_normal(FILTER_SIZE, IMAGE_DEPTH, stddev=1.0)
        bias1[i] = 0
     
    for i in range(0, NUM_FILT2):
        filt2[i] = _initialize_truncated_normal(FILTER_SIZE, NUM_FILT1, stddev=1.0)
        bias2[i] = 0
     
    w1 = IMAGE_WIDTH - FILTER_SIZE + 1
    w2 = w1 - FILTER_SIZE + 1

    theta3 = 0.01 * np.random.rand(NUM_OUTPUT_NODES, (w2 / 2) * (w2 / 2) * NUM_FILT2)

    bias3 = np.zeros((NUM_OUTPUT_NODES, 1))

    return [filt1, filt2, bias1, bias2, theta3, bias3]

def _initialize_truncated_normal(FILTER_SIZE, IMG_DEPTH, stddev=1.0):
    lower = -1
    upper = 1

    fan_in = FILTER_SIZE * FILTER_SIZE * IMG_DEPTH
    stddev = stddev * np.sqrt(1. / fan_in)
    shape = (IMG_DEPTH, FILTER_SIZE, FILTER_SIZE)
    
    return scipy.stats.truncnorm.rvs(
          (lower)/stddev, (upper)/stddev, loc=0, scale=stddev, size=shape)
