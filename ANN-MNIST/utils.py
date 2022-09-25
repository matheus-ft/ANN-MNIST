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


def cross_entropy(y: np.matrix) -> float:
    pass
