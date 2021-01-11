# Simple activation functions 
import numpy as np

# rectified linear activation (ReLU)
def relu(x):
    if x > 0:
        return x
    else:
        return 0

# leaky relu
def leaky_relu(x,small_number):
    if x > 0:
        return x
    else:
        return x * small_number

# logistic sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

