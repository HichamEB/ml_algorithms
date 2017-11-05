from model import Lasso

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def main():

    np.random.seed(0)

    n_samples = 100

    # Generate data
    X = np.linspace(0, 20, n_samples)
    np.random.shuffle(X)

    noise = 0.1 * np.random.normal(size=(n_samples,))
    y = np.sin((np.pi / 10.) * X)
    y_noisy = y + noise

    # Add polynomial features
    X_polynomial = np.concatenate([X.reshape(n_samples, -1),
                                   X.reshape(n_samples, -1) ** 2,
                                   X.reshape(n_samples, -1) ** 3], axis=1)

    # Model fitting
    regularization = 10.
    rho = 100.
    n_iterations = 10

    model = Lasso(regularization, rho)
    model.train(X_polynomial, y, n_iterations)

    print("OLS weights :\n{}".format(model.weights[1:, :]))
    print("OLS bias :\n{}".format(model.weights[0, :]))

    # Model prediction
    y_predicted = model.predict(X_polynomial)

    # Plots
    alpha = 0.5
    plt.scatter(X, y, label="ground_truth", alpha=alpha * 2)
    plt.scatter(X, y_noisy, label="signal", alpha=alpha)
    plt.scatter(X, y_predicted, label="model prediction", alpha=alpha)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
