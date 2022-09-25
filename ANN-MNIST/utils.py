import numpy as np
from scipy.special import expit  # np.exp is slower and math.exp can't receive array


def sigmoid(z: float | np.ndarray):
    """Gives the sigmoid logistic function value on z.

    Parameters
    ----------
    z : float | np.ndarray
        Argument, or array of arguments, for the logistic function

    Returns
    -------
    float | np.ndarray
        Returns the numerical value of the logistic function, or an np.ndarray of such values
    """
    return expit(z)


def sigmoid_prime(z: float | np.ndarray):
    """Gives the derivative of the sigmoid logistic function g:

    g'(z) = g(z) * [1 - g(z)]

    Parameters
    ----------
    z : float | np.ndarray
        Argument, or array of arguments, for the derivative of the logistic function

    Returns
    -------
    float | np.ndarray
        Returns the numerical value of the derivative of the logistic function, or an np.ndarray of such values
    """
    a = expit(z)
    return a * (1 - a)


def _binary_cross_entropy(label: np.ndarray, prediction: np.ndarray) -> float:
    if len(label.shape) > 1 or len(prediction.shape) > 1:
        raise Exception("Use the funtion `cross_entropy` if there's more than 1 class")

    m = label.shape[0]  # number of examples
    odd_yes = label * np.log(prediction)
    odd_no = (1 - label) * np.log(1 - prediction)
    costs = odd_yes + odd_no
    J = sum(costs)
    return -J / m


def cross_entropy(
    label: np.ndarray | np.matrix, prediction: np.ndarray | np.matrix
) -> float:
    if len(label.shape) == 1 == len(prediction.shape):
        return _binary_cross_entropy(label, prediction)
    elif label.shape != prediction.shape:
        raise Exception("`label` and `prediction` must have the same dimensions")

    m = label.shape[0]  # number of examples
    K = label.shape[1]  # number of classes
    J = 0
    for k in range(K):
        odd_yes = label[:, k] * np.log(prediction[:, k])
        odd_no = (1 - label[:, k]) * np.log(1 - prediction[:, k])
        costs = odd_yes + odd_no
        J += sum(costs)
    return -J / m
