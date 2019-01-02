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


class LinearRegression:
    """
    A class to run and analyze a Linear Regression
    TODO: P-Value
    """

    def __init__(self, a, b):
        """
        Run a Linear Regression

        :param a: input matrix
        :param b: output matrix
        """

        # Record input/output matrices, add bias column
        self.A = np.pad(a, ((0, 0), (0, 1)), mode='constant', constant_values=1.0)
        self.b = b

        """
        Need to solve for x:
        A • x = b                   

        Can only eliminate A by taking the dot product with inverse(A), if we know A is square.
        We must make A square, which we can do with its transpose:
        (A' • A) • x = A' • b

        Now we can use the inverse:
        inverse(A' • A) • (A' • A) • x = inverse(A' • A) • A' • b

        And simplify:
        x = inverse(A' • A) • A' • b
        """
        inverse_chunk = np.dot(a.T, a)
        inverse_chunk = np.linalg.inv(inverse_chunk)
        before_b = np.dot(inverse_chunk, a.T)
        self.coefficients = np.dot(before_b, b)

        # Calculated Observation
        self.y_hat = np.dot(a, self.coefficients)
        self.observations = len(a)

        # Calculating some variables for model evaluation
        ssr = np.sum(np.square(self.y_hat - np.mean(b)))
        sse = np.sum(np.square(b - self.y_hat))
        self.r_squared = round(float(ssr) / float(ssr + sse), 4)
        self.standard_error = np.sqrt(sse / 4)
        self.accuracy = round(float(self.standard_error / np.mean(b)), 3)

    def predict(self, a):
        """
        Returns a prediction based on the model coefficients

        :param a: Input matrix
        :return: The forecast of the linear regression
        """
        return np.dot(a, self.coefficients)

