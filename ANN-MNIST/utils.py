import numpy as np
from scipy.special import expit as sigmoid  # faster than using np.exp


class vector(np.ndarray):
    """2-dimensional numpy array representing a column vector.

    Extends the numpy.ndarray class to be able to treat arrays as "proper" vector in R^n.

    Parameters
    ----------
    lst : list[float]
        iterable from which the vector is created
    """

    def __new__(cls, lst: list[float]):
        v = np.array([lst]).reshape(len(lst), 1)  # the value that actually matters
        v = np.asarray(v).view(cls)  # to make it behave as its own "data type"
        return v


class matrix(np.ndarray):
    """2-dimensional numpy array representing a matrix.

    Extends the numpy.ndarray class to be able to build a matrix either by a list of lists or by a list of column vectors.

    Parameters
    ----------
    lst : list[list[float]] | list[vector]
        list containing list[float]s or vectors of floats
    """

    def __new__(cls, lst: list[list[float]] | list[vector]):
        m = np.array(lst)
        if type(lst[0]) == vector:
            m = np.concatenate(m, axis=1)
        m = np.asarray(m).view(cls)  # to make it behave as its own "data type"
        return m


def sigmoid_prime(z: float) -> float:
    """Gives the derivative of the sigmoid logistic function g.

    g'(z) = g(z) * [1 - g(z)]

    Parameters
    ----------
    z : float
        Argument passed to the function. Can be an np.ndarray of floats and will operate coordinate-wise.

    Returns
    -------
    float
        Returns the numerical value of the derivative of the logistic function, or an np.ndarray of such values
    """
    a = sigmoid(z)
    return a * (1 - a)
