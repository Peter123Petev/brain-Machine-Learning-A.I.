
## brain  
  
##### Peter Petev's Custom Machine Learning / Artificial Intelligence Python Package  
***  
"brain" is my attempt to merge all of my Machine Learning / Artificial Intelligence code into one easy to use package, which outlines the math of the algorithms.  
  
Writing code for this package has helped me understand some of the more difficult algorithms associated with M.L./A.I., and it has allowed me to develop custom applications much faster.  
  
Can be used for:  
* Neural Networks:  
    ```python  
	 import numpy as np import brain
	 
	 # Input Matrix, all inputs from 0 to 1 (bias built in)
	 X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
	 
	 # Output Matrix, all outputs from 0 to 1
	 Y = np.array([[1], [1], [0], [0]])
	 
	 # Create Neural Network
	 inputs = X                     # Input Matrix
	 layers = [100, 100, 100]       # Hidden Layer Sizes
	 outputs = Y                    # Output Matrix
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
	 ```
	 ```
	 Console:
	 Max Error: 4.646613045322656e-05
	 Prediction: [[3.79146463e-05]]
	 ```
* Regression:  
	```python  
	 import numpy as np import brain
	 
	 # Input Matrix, all inputs from 0 to 1 (bias built in)
	 X = np.array([[1, 2, 3], [3, 4, 3], [1, 3, 7], [4, 9, 4]])
	 
	 # Output Matrix, all outputs from 0 to 1
	 Y = np.array([[7], [9], [10], [18]])
	 
	 # Do Linear Regression
	 example = brain.LinearRegression(X, Y)
	 
	 # Some Evaluation metrics (no p-value yet)
	 print("R Squared:", example.r_squared)
	 print("Accuracy:", example.accuracy, "< 0.2?")
	 
	 # Coefficients
	 print("Coefficients:\n", example.coefficients)  
	 
     # To Predict a Point
     print("Prediction:\n", example.predict(X))
     ```
	```
	Console:
	R Squared: 0.9768
	Accuracy: 0.063 < 0.2?
	Coefficients:
	[[0.20489174]
	[1.56134723]
	[0.79190056]]
	Prediction:
	[[ 5.70328789]
	[ 9.23576584]
	[10.43223737]
	[18.03929431]]
	```
* (W.I.P.) Genetic Algos:  
* (W.I.P.) Probability:  
*