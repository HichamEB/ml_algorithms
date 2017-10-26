#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class OrdinaryLeastSquares(object):
    def __init__(self, learning_rate=1e-6):
        self.fitted = False
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return (1./y_pred.shape[0]) * np.linalg.norm(y_true.reshape(-1, 1) - y_pred) ** 2

    def train(self, X, y, epochs=1000, verbose=False):
        assert len(X.shape) == 2
        assert len(y.shape) <= 2

        if not self.bias:
            self.bias = np.random.normal()

        if not self.weights:
            self.weights = np.random.normal(size=(X.shape[1], 1))

        self.fitted = True

        prevision = self.predict(X)
        if verbose:
            error = self.mean_squared_error(y, prevision)
            print("MSE = {}".format(error))

        for _ in range(epochs):

            weights_update = np.dot(X.T, (prevision - y.reshape(-1, 1)))
            bias_update = np.dot(np.ones(shape=(X.shape[0], 1)).T, (prevision - y.reshape(-1, 1)))

            self.weights += - weights_update * self.learning_rate
            self.bias = - bias_update * self.learning_rate

            prevision = self.predict(X)
            if verbose:
                error = self.mean_squared_error(y, prevision)
                print("MSE = {}".format(error))

    def predict(self, X):
        assert self.fitted
        return np.dot(X, self.weights) + self.bias * np.ones(shape=(X.shape[0], 1))
