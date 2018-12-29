import numpy as np
import brain

# Input Matrix, all inputs from 0 to 1 (bias built in)
X = np.array([[1, 2, 3],
              [3, 4, 3],
              [1, 3, 7],
              [4, 9, 4]])

# Output Matrix, all outputs from 0 to 1
Y = np.array([[7],
              [9],
              [10],
              [18]])

# Do Linear Regression
example = brain.LinearRegression(X, Y)

# Some Evaluation metrics (no p-value yet)
print("R Squared:", example.r_squared)
print("Accuracy:", example.accuracy, "< 0.2?")

# Coefficients
print("Coefficients:\n", example.coefficients)

# To Predict a Point
print("Prediction:\n", example.predict(X))


