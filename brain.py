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

from brain_methods import linearregression, neuralnetwork, guessformula


def NeuralNetwork(x, layers, y):
    return neuralnetwork.NeuralNetwork(x, layers, y)


def LinearRegression(x, y):
    return linearregression.LinearRegression(x, y)


def GuessFormula(x, y):
    return guessformula.GuessFormula(x.tolist(), y.tolist())

