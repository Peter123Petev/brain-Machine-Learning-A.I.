import numpy as np
import brain

# Input Matrix, all inputs from 0 to 1 (bias built in)
X = np.array([[1, 0],
              [0, 1],
              [1, 1],
              [0, 0]])

# Output Matrix, all outputs from 0 to 1
Y = np.array([[1],
              [1],
              [0],
              [0]])

# nn = brain.NeuralNetwork(X, [100, 100, 100], Y, trials=10000)
# print(nn.y_hat)
# print(np.mean(nn.y_hat - Y))

# # Do Linear Regression
# example = brain.LinearRegression(X, Y)
#
# # Some Evaluation metrics (no p-value yet)
# print("R Squared:", example.r_squared)
# print("Accuracy:", example.accuracy, "< 0.2?")
#
# # Coefficients
# print("Coefficients:\n", example.coefficients)
#
# # To Predict a Point
# print("Prediction:\n", example.predict(X))


