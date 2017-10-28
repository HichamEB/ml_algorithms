#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class OrdinaryLeastSquares(object):
    def __init__(self):
        self.fitted = False
        self.weights = None
        self.bias = None

    def train(self, X, y):
        assert len(X.shape) == 2
        assert len(y.shape) <= 2

        X_tilde = np.append(np.ones((X.shape[0], 1)), X, axis=1)

        weights = np.dot(X_tilde.T, X_tilde)
        weights = np.linalg.inv(weights)
        weights = np.dot(weights, X_tilde.T)
        weights = np.dot(weights, y.reshape(X.shape[0], -1))

        self.bias = weights[0, :]
        self.weights = weights[1:, :]
        self.fitted = True

    def predict(self, X):
        assert self.fitted

        prevision = np.dot(X, self.weights) + self.bias * np.ones((X.shape[0], 1))
        return prevision.reshape(-1)
