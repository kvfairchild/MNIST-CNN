## MNIST CNN
From-scratch CNN with two convolutional layers, two max pooling layers, one fully connected layer, and AdaGrad optimizer.

* main.py: main file to init and run network  
* config.py: set net and training parameters (filter dimensions, learning rate, etc.)  
* MNIST_file_parser.py: parse MNIST files  
* initialize.py: initalize network (truncated normal distribution)  
* run.py: train and test network on MNIST dataset  
* network.py: CNN (feedforward & backprop)  
* optimize.py: contains AdaGrad optimizer  

Typical training output:

    Batch 50/500 of Epoch 1/1: Cost: 0.35, Batch: 91% accuracy, Epoch: 85% accuracy

Typical test output:

    #50/10000: 7 | 7  OK
    #51/10000: 8 | 3  X


To run:  
> ./main.py
