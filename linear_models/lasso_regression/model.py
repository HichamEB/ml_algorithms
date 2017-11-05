import numpy as np


def soft_threshold(x, threshold):
    if x > threshold:
        return 1.
    elif x < - threshold:
        return -1.
    else:
        return 0.


class Lasso(object):

    def __init__(self, regularization=1., rho=1.):
        self.regularization = regularization
        self.rho = rho

        self.fitted = False
        self.weights = None

    def train(self, X, y, n_iterations=10):
        assert len(X.shape) == 2
        assert len(y.shape) <= 2

        n_samples = X.shape[0]
        n_features = X.shape[1] + 1
        ones = np.ones((n_samples, 1))
        X_tilde = np.append(ones, X, axis=1)

        a = np.random.normal(size=(n_features, 1))
        b = np.random.normal(size=(n_features, 1))

        inv = (np.dot(X_tilde.T, X_tilde) +
               self.rho * np.identity(n_features))

        inv = np.linalg.inv(inv)

        for _ in range(n_iterations):

            x = np.dot(inv, (np.dot(X_tilde.T, y.reshape(n_samples, -1)) +
                             self.rho * a - b))

            a = (x + b / self.rho)
            a = np.apply_along_axis(
                lambda v: soft_threshold(v, self.regularization / self.rho),
                1, a
            ).reshape(b.shape)

            b += self.rho * (x - a)

        weights = np.dot(inv, (np.dot(X_tilde.T, y.reshape(n_samples, -1)) +
                               self.rho * a - b))

        self.weights = weights
        self.fitted = True

    def predict(self, X):
        assert self.fitted

        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1))
        X_tilde = np.append(ones, X, axis=1)

        prevision = np.dot(X_tilde, self.weights)

        return prevision.reshape(-1)
