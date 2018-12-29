#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Peter Petev"
__copyright__ = "Copyright 12/29/2018"
__credits__ = ["Peter Petev"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Peter Petev"
__email__ = "pspetev@gmail.com"
__status__ = "Development"

import numpy as np


class NeuralNetwork:
    """
    A class to represent a simple Neural Network
    """
    def __init__(self, x, inner_layers, y, trials=0):
        """
        Initialize the Neural Network with inputs, outputs, and the sizes of hidden layers.

        :param x: input matrix
        :param inner_layers: list of hidden layer sizes
        :param y: output matrix
        :param trials: number of trials to be run on instantiation call
        """
        # Record input/output matrices, add bias column
        self.x = np.pad(x, ((0, 0), (0, 1)), mode='constant', constant_values=1.0)
        self.y = y

        # Combine layer sizes into one list
        self.layer_sizes = inner_layers
        self.layer_sizes.insert(0, len(self.x[0]))
        self.layer_sizes.append(len(self.y[0]))

        # Create random synapse matrices which map each adjacent layer
        self.synapses = []
        for i in range(0, len(self.layer_sizes) - 1):
            self.synapses.append(2 * np.random.random((self.layer_sizes[i], self.layer_sizes[i + 1])) - 1)

        # Optionally run trials
        self.y_hat = None
        self.train(trials)

    def train(self, trials):
        for trial in range(0, trials):
            # Create Input layer, add to a layer list
            layers = []
            layers.append(self.x)
            """
            Use dot product of a layer and its synapse to create next layer:
                Layer 0: Inputs 
                Layer 1: Inputs * Synapse 0
                ...
                Layer (n): Layer (n-1) * Synapse (n-1)
                Layer Final: y hat
            """
            for i in range(0, len(self.layer_sizes) - 1):
                layers.append(self.sigmoid(np.dot(layers[i], self.synapses[i])))

            """
            Back-propagate to distribute error:
                Delta (Final) =         ( y layer - y hat layer ) * ∇Cost(y layer)
                    Synapse Error (Penultimate) = Delta (final) • Transpose Synapse (to Penultimate)
                Delta (Penultimate) =   Synapse Error (Penultimate) * ∇Cost( Layer (Penultimate) )
                    Synapse Error (n - 1) = Delta (n) • Transpose Synapse (to n-1)
                Delta (n - 1) =         Synapse Error (n - 1) * ∇Cost( Layer (n - 1) )
                ...
            """
            deltas = []
            # Hammond product between (y - y hat) matrix and (derivative cost function of y) matrix, add to a delta list
            deltas.insert(0, np.multiply((self.y - layers[len(self.layer_sizes) - 1]), self.sigmoid(layers[len(self.layer_sizes) - 1], derivative=True)))
            for i in range(0, len(self.synapses) - 1):
                # Find index of previous layer, and the bridging synapse has the same index
                # Remember: Layer (t) = Layer (t-1) • Synapse (t-1)
                syn_index = len(self.synapses) - 1 - i

                # Distribute error = the dot product of the next layer delta and the transposed synapse to it
                synapse_error = deltas[0].dot(self.synapses[syn_index].T)

                # Hammond product between distributed_error and each layer to find its delta, put at front of delta list
                deltas.insert(0, np.multiply(synapse_error, self.sigmoid(layers[syn_index], derivative=True)))

            """
            Use deltas to adjust weights:
                Synapse (n) = Transpose Layer (n) • Deltas (n)
            """
            for i in range(0, len(self.synapses)):
                self.synapses[i] += layers[i].T.dot(deltas[i])

            # Track Most Recent y hat for debugging
            self.y_hat = layers[len(layers) - 1]

    def predict(self, x):
        """
        Predicts output for an input based on the Neural Network

        :param x: Set of inputs, each [0, 1]
        :return: Predicted Output [0, 1]
        """
        prediction = np.pad(x, ((0, 0), (0, 1)), mode='constant', constant_values=1.0)

        # Use dot product through the synapse matrices to get to a prediction
        for synapse in self.synapses:
            prediction = self.sigmoid(np.dot(prediction, synapse))
        return prediction

    def sigmoid(self, a, derivative=False):
        """
        Popular cost function for Neural Networks

        :param a: Object to have all of its elements run through the sigmoid function
        :param derivative: Boolean describing which form of the equation to use
        :return: Object a with all elements being run through the sigmoid function (or its derivative)
        """
        if derivative:
            return self.sigmoid(a) * (1 - self.sigmoid(a))
        return 1 / (1 + np.round(np.exp(-a), 20))

