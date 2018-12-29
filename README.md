## brain

##### Peter Petev's Custom Machine Learning / Artificial Intelligence Python Package
***
"brain" is my attempt to merge all of my Machine Learning / Artificial Intelligence code into one easy to use package, which outlines the math of the algorithms.

Writing code for this package has helped me understand some of the more difficult algorithms associated with M.L./A.I., and it has allowed me to develop custom applications much faster.

Can be used for:
* Neural Networks:
	```python
	import numpy as np  
	import brain  
  
	# Input Matrix, all inputs from 0 to 1 (bias built in)
	X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  
	# Output Matrix, all outputs from 0 to 1
	Y = np.array([[0], [1], [1], [0]])
	
	# Create Neural Network    
	inputs = X			# Input Matrix
	layers = [5, 5]		# Hidden Layer Sizes
	outputs = Y			# Output Matrix
	example = brain.NeuralNetwork(inputs, layers, outputs)  
  
	# Train the Neural Network 10000 times
	example.train(10000)  
  
	# Print most recent prediction  
	print(example.y_hat)  
  
	# Predict a new point  
	forecast = example.predict([[0, 1]])  
	print("Prediction:", forecast)
	```
	```
	Console:
	[[0.00184528]
	[0.99713676]
	[0.99772455]
	[0.00213567]]
	Prediction: [[0.9971367]]
	```
* (W.I.P.) Regression:
* (W.I.P.) Genetic Algos:
* (W.I.P.) Probability:
* 