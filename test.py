import numpy as np
import brain

# Input Matrix, all inputs from 0 to 1 (bias built in)
X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

# Output Matrix, all outputs from 0 to 1
Y = np.array([[1], [1], [0], [0]])

# Create the Guesses
example = brain.GuessFormula(X, Y)

# Fit Guesses to Data
example.fit(0.99)

# Print best guest
example.best_guess()

# Print a new point
forecast = example.predict(X)
print("Prediction:\n", forecast)

