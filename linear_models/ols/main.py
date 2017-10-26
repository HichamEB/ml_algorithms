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

    model = OrdinaryLeastSquares()  # OLS trained with Gradient Descent
    model.train(X.reshape(-1, 1), y, verbose=True)
    print("OLS weights : {}".format(model.weights))
    print("OLS bias : {}".format(model.bias))

    y_predicted = model.predict(X.reshape(-1, 1))

    plt.scatter(X, y, label="ground_truth", alpha=0.5)
    plt.scatter(X, y_noisy, label="signal", alpha=0.5)
    plt.scatter(X, y_predicted, label="model prediction", alpha=0.5)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
