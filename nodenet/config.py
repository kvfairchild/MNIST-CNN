# MNIST CNN SETTINGS

# data specs (MNIST)
IMAGE_WIDTH = 28 # 28 x 28 square images (w = h)
IMAGE_DEPTH = 1

# output nodes (MNIST)
NUM_OUTPUT_NODES = 10 # digits 0-9

# convolution params
FILTER_SIZE = 5
NUM_FILT1 = 32
NUM_FILT2 = 16

# training params
LEARNING_RATE = 0.1
NUM_EPOCHS = 1
BATCH_SIZE = 100
DISPLAY_STEP = 10
