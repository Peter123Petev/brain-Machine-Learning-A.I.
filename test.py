import numpy as np
import brain

# Input Matrix, all inputs from 0 to 1 (bias built in)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output Matrix, all outputs from 0 to 1
Y = np.array([[0],
              [1],
              [1],
              [0]])

# Create Neural Network in one line
#
#   Parameters:
#   - inputs: input matrix
#   - layers: hidden layer sizes
#   - outputs: output matrix
#   - trials: (optional) number of trials to train
#
#   Returns:
#   - NeuralNetwork Object
inputs = X
layers = [5, 5]
outputs = Y
example = brain.NeuralNetwork(inputs, layers, outputs)

# Train the Neural Network
trials = 10000
example.train(trials)

# Print most recent prediction
print(example.y_hat)

# Predict a new point
forecast = example.predict([[0, 1]])
print("Point Estimate:", forecast)
