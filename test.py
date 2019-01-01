import numpy as np
import brain

# Input Matrix, all inputs from 0 to 1 (bias built in)
X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

# Output Matrix, all outputs from 0 to 1
Y = np.array([[1], [1], [0], [0]])

# Create Neural Network
inputs = X                      # Input Matrix
layers = [100, 100, 100]        # Hidden Layer Sizes
outputs = Y                     # Output Matrix
example = brain.NeuralNetwork(X, [100, 100, 100], Y)

# Train the neural network 10000 times
example.train(10000)

# Train the neural network until maximum error
example.fit(0.0001)

# Print Max Model Error
print("Max Error:", example.error)

# Predict a new point
forecast = example.predict([[0, 0]])
print("Prediction:", forecast)

