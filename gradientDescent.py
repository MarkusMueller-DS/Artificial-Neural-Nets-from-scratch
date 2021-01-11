# Gradient Descent in action

# imports 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# start weight, goal prediction and input value
weight, goal_pred, input_ = (0.5, 0.8, 0.5)

# empty arrays to catch weight and error per epoch
weight_array = []
error_array = []

for iteration in range(10):
    # append weights to array
    weight_array.append(weight)
    pred = input_ * weight
    error = (pred - goal_pred) ** 2
    # append error to error array
    error_array.append(error)
    delta = pred - goal_pred
    weight_delta = delta * input_
    weight = weight - weight_delta


weight_array = np.array(weight_array)
error_array = np.array(error_array)

# visualize error and weights

# values for weights between a specific range
x = np.linspace(0.5,2.7,20)

# function for y (error)
# error = ((input * weight) - goal_pred) ** 2
y = ((input_ * x) - goal_pred)** 2

plt.plot(x,y);
plt.scatter(weight_array, error_array);
plt.xlabel ('weight');
plt.ylabel ('error');
# plt.savefig('test.png', dpi=150);
plt.show();

