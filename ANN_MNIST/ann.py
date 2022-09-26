import numpy as np
import numpy.random as npr
from . import utils as ut


class model_hyperparams:
    def __init__(
        self,
        activation_function=ut.sigmoid,
        loss_function=ut.cross_entropy,
        use_xavier: bool = True,
    ) -> None:
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.use_xavier = use_xavier


class ANN:
    def __init__(
        self,
        n_classes: int,
        n_neurons_per_layer: int,
        n_hidden_layers: int = 1,
        model=model_hyperparams(),
        regularization_lambda: float = 0,
    ) -> None:
        ### Architecture
        self.__number_classes = n_classes
        self.__number_hidden_layers = n_hidden_layers
        self.__neurons_per_layer: list[int] = self._neurons_list(n_neurons_per_layer)
        self.__activation_function = model.activation_function
        self.__use_xavier: bool = model.use_xavier
        ### Data
        self._weights: list[np.matrix] = self._initialize_weights()
        self._activation: list[np.matrix] = None
        self._gradient: list[np.matrix] = [
            np.matrix(np.zeros(theta.shape)) for theta in self.weights
        ]
        ### Hyperparameters
        self.regularization_lambda: float = regularization_lambda

    @property
    def number_classes(self) -> int:
        return self.__number_classes

    @property
    def number_hidden_layers(self) -> int:
        return self.__number_hidden_layers

    @property
    def neurons_per_layer(self) -> list[int]:
        return self.__neurons_per_layer

    @property
    def activation_function(self):
        return self.__activation_function

    @property
    def use_xavier(self) -> bool:
        return self.__use_xavier

    @property
    def weights(self) -> list[np.matrix]:
        return self._weights

    @property
    def activation(self) -> list[np.matrix]:
        return self._activation

    @property
    def gradient(self) -> list[np.matrix]:
        return self._gradient

    def _neurons_list(self, n_neurons_per_layer: int) -> list[int]:
        neurons_list = [n_neurons_per_layer] * self.number_hidden_layers
        neurons_list.append(self.number_classes)
        return neurons_list

    @staticmethod
    def _random_matrix_xavier(rows: int, columns: int) -> np.matrix:
        epislon = (6 / (rows + columns - 1)) ** 0.5
        arr = npr.rand(rows, columns)  # uniform distribution
        return np.matrix(arr * 2 * epislon - epislon)

    @staticmethod
    def _random_matrix_he(rows: int, columns: int) -> np.matrix:
        sigma = (2 / (rows + columns - 1)) ** 0.5
        arr = npr.randn(rows, columns)  # normal distribution
        return np.matrix(arr * sigma)

    def _random_matrix(self, rows: int, columns: int) -> np.matrix:
        if self.use_xavier:
            return self._random_matrix_xavier(rows, columns)
        else:
            return self._random_matrix_he(rows, columns)

    def _initialize_weights(self) -> list[np.matrix]:
        weights = []
        for i in range(1, self.number_hidden_layers + 1):
            neurons_left = self.neurons_per_layer[i - 1] + 1  # +1 for bias
            neurons_right = self.neurons_per_layer[i]
            random_theta = self._random_matrix(neurons_right, neurons_left)
            weights.append(random_theta)
        return weights

    @staticmethod
    def add_column_1s(data_matrix: np.matrix) -> np.matrix:
        """Add a column of 1s to left of the data given.

        Parameters
        ----------
        array_like : np.matrix | ut.vector
            Iterable on which the column of 1s is added

        Returns
        -------
        np.matrix | ut.vector
            Same iterable with the additional column of 1s to the left
        """
        m = data_matrix.shape[0]
        ones = np.ones((m, 1))
        array = np.concatenate((ones, data_matrix), axis=1)
        return np.matrix(array)

    def _forward_pass(self, examples: np.matrix) -> None:
        m = examples.shape[0]  # number of examples
        n = examples.shape[1]  # number of features
        if len(self.weights) == self.number_hidden_layers:
            self._weights.insert(  # meaning the bias is missing
                0, self._random_matrix(self.neurons_per_layer[0], n + 1)
            )
        if self.activation is None:
            self._activation = [
                np.matrix(np.zeros((m, n_l))) for n_l in self.neurons_per_layer
            ]
        ex = self.add_column_1s(examples)
        z_0 = ex @ self.weights[0].T
        self._activation[0] = self.activation_function(z_0)
        for j in range(1, self.number_hidden_layers + 1):
            a_j_1 = self.add_column_1s(self.activation[j - 1])
            z_j = a_j_1 @ self.weights[j].T
            self._activation[j] = self.activation_function(z_j)

    def _backward_pass(self, labels: np.matrix) -> list[np.matrix]:
        m = labels.shape[0]  # number of examples
        H = self.number_hidden_layers
        prediction = self.activation[H]  # last activation, i.e. output of the net
        delta_L = prediction - labels
        self._gradient[H] = prediction @ delta_L.T
        for L in range(H - 1, -1, -1):  # goes to 0 bc the training data is not included
            a_L = self.activation[L]
            a_L = self.add_column_1s(a_L)  # needed only to match dimensions
            theta_L = self.weights[L + 1]
            delta_L = delta_L.T @ theta_L * a_L * (1 - a_L)
            delta_L = delta_L[:, 1:]  # discarding the "delta" of the bias "feature"
            D = a_L @ delta_L.T
            self._gradient[L] = D / m
        return self._regularized_gradient()

    def _backpropagation(
        self, examples: np.matrix, labels: np.matrix
    ) -> list[np.matrix]:
        self._forward_pass(examples)
        return self._backward_pass(labels)

    def _regularized_gradient(self) -> list[np.matrix]:
        _lambda = self.regularization_lambda
        for L in range(self.number_hidden_layers + 1):
            gradient = self.gradient[L]
            m = gradient.shape[0]  # number of examples
            regularized_theta = _lambda * self.weights[L] / m
            regularized_theta[:, 0] *= m / _lambda  # not regularizing the bias
            gradient += regularized_theta
            self._gradient[L] = np.matrix(gradient)
        return self.gradient
