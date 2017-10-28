#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from model import OrdinaryLeastSquares


def main():

    np.random.seed(0)

    n_samples = 1000

    X = np.linspace(0, 50, n_samples)
    np.random.shuffle(X)

    y = np.sin((np.pi/10.) * X)

    noise = np.random.normal(size=(n_samples,))
    y_noisy = y + noise

    X_polynomial = np.append(X.reshape(n_samples, -1), X.reshape(n_samples, -1)**2, axis=1)
    X_polynomial = np.append(X_polynomial, X.reshape(n_samples, -1) ** 3, axis=1)

    model = OrdinaryLeastSquares()
    # model.train(X.reshape(n_samples, -1), y)
    model.train(X_polynomial, y)
    print("OLS weights : {}".format(model.weights))
    print("OLS bias : {}".format(model.bias))

    # y_predicted = model.predict(X.reshape(n_samples, -1))
    y_predicted = model.predict(X_polynomial)

    plt.scatter(X, y, label="ground_truth", alpha=0.5)
    plt.scatter(X, y_noisy, label="signal", alpha=0.5)
    plt.scatter(X, y_predicted, label="model prediction", alpha=0.5)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
