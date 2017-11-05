import numpy as np


class Ridge(object):

    def __init__(self, regularization=1.):
        self.regularization = regularization
        self.fitted = False
        self.weights = None

    def train(self, X, y):
        assert len(X.shape) == 2
        assert len(y.shape) <= 2

        n_samples = X.shape[0]
        n_features = X.shape[1] + 1
        ones = np.ones((n_samples, 1))
        X_tilde = np.append(ones, X, axis=1)

        weights = (np.dot(X_tilde.T, X_tilde) +
                   self.regularization * np.identity(n_features))
        weights = np.linalg.inv(weights)
        weights = np.dot(weights, X_tilde.T)
        weights = np.dot(weights, y.reshape(n_samples, -1))

        self.weights = weights
        self.fitted = True

    def predict(self, X):
        assert self.fitted

        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1))
        X_tilde = np.append(ones, X, axis=1)

        prevision = np.dot(X_tilde, self.weights)

        return prevision.reshape(-1)
