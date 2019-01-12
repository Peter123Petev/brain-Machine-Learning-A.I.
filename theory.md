# Theory
Some theory behind Machine Learning / A.I.
---
* [Linear Regression](#mlr)
* [Artificial Neural Network](#ann)
---
#### Linear Regression<a name="mlr"></a>
A: Input Data Matrix (rows as data inputs with columns as variables)
b: Output Data Matrix (rows as data outputs)
x: Coefficient Matrix

Goal: Find x with least error to predict b
Initial Formula:
<center>Ax = b</center>

Multiply both sides by A<sup>T</sup> on left:
<center>(A<sup>T</sup>)Ax = (A<sup>T</sup>)b &rarr; (A<sup>T</sup>A)x = (A<sup>T</sup>)b</center>

This creates a matrix (A<sup>T</sup>A), which is a square matrix regardless of A's dimensions. Now multiply (A<sup>T</sup>A) by its inverse on the left:
<center>(A<sup>T</sup>A)<sup>-1</sup>(A<sup>T</sup>A)x = (A<sup>T</sup>A)<sup>-1</sup>(A<sup>T</sup>)b</center>

An inverse matrix multiplied by its inverse is the identity matrix, so the previous equation simplifies to:
<center>x = (A<sup>T</sup>A)<sup>-1</sup>(A<sup>T</sup>)b</center>

#### Artificial Neural Network<a name="ann"></a>
A: Input Data Matrix (rows as data inputs with columns as variables)
b: Output Data Matrix (rows as data outputs)
w: Weight Matrix

Goal: Find x through gradient descent to predict b

In this case, we will model each layer after the inputs by the previous layer dot the weight matrix to the next layer, then passed through an activation function (function which turns the output into a number between 0 and 1).
<center>Input Layer = A</center>
<center>Next Layer = activation(A dot w)<center>


We want to define a cost function to minimize. In this case, we can use:
<center>Cost = (1 / 2) · sum of [(y - activation(Final Layer))<sup>2</sup>]

To use gradient descent, we need to find the steepest descent. In functions, the gradient (∇) of a function is its steepest ascent, so in this case we would have:
<center>∇ Cost  = (1 / 2) · (2) · (-activation derivative(Final Layer)) · sum of (y - activation(Final Layer))</center>
Which simplifies to:
<center>∇ Cost  = -activation derivative(Final Layer) · Final Layer Error</center>

But we need steepest descent, not steepest ascent. So in this case we just multiply the gradient function by -1:
<center>Steepest descent = - ∇ Cost  = activation derivative(Final Layer) · Final Layer Error</center>

Now, to backpropogate the error (basically tell each previous layer how much it had, since we only know final layer's error) we need to multiply each layer's error by previous weight matrix to get the previous layer's error:
<center>Previous Layer Error = Current Layer Error * w<sup>T</sup></center>
<center>(Notice the multiplication by element, not matrix dot product)</center>

To adjust each of the weight matrices to have the least error, you need to do gradient descent. You can do this by calculating the steepest descent for each layer's weight matrix, then adding it to the weight matrix:
<center> w = w + steepest descent function of layer created by weight</center>

After doing this many, many times, we can say that we have most likely settled into a local minimum of the cost function.