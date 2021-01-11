# code from Theory of Neural Networks - DL without Frameworks 
import numpy as np

# start with random weights
weights = [0.4, -0.8, 0.2]
start_weights = [0.4, -0.8, 0.2]
alpha = 0.1

streetlights = np.array([[0,0,1],
                         [0,1,1],
                         [0,0,1],
                         [1,1,1],
                         [0,1,1],
                         [1,0,1]])

walk_vs_stop = np.array([0,1,0,1,1,0])

# iteration
for i in range(20):
    for row_index in range (len(walk_vs_stop)):
        # assign datapoints for each row
        input_ = streetlights[row_index]
        goal_pred = walk_vs_stop[row_index]
        # dot product of input values and the weights 
        pred = input_.dot(weights) 
        # square the error as a KPI
        error = (pred - goal_pred) ** 2
        # calculate the delta between pred and actual value
        delta = pred - goal_pred
        # update the weights
        weights = weights - (alpha * (input_* delta))
        # print the predictions for each row
        print("Prediction: " + str(pred))

    print("Start_weights: "+ str(start_weights))
    print("Weights: " + str(weights))
    print("Error: " + str(error))
    print("-"*30)
