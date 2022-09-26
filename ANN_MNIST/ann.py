import numpy as np
import numpy.random as npr
from . import utils as ut


class training_hyperparams:
    def __init__(
        self,
        learning_rate: float,
        number_iterations: int,
        regularization_lambda: float = 0,
        optimizer=ut.gradient_descent,
    ) -> None:
        self.learning_rate: float = learning_rate
        self.number_iterations: int = number_iterations
        self.regularization_lambda: float = regularization_lambda
        self.optimizer = optimizer


class ANN:
    def __init__(
        self,
        n_classes: int,
        n_neurons_per_layer: int,
        n_hidden_layers: int = 1,
    ) -> None:
        ### Architecture
        self.__number_classes = n_classes
        self.__number_hidden_layers = n_hidden_layers
        self.__neurons_per_layer: list[int] = self._neurons_list(n_neurons_per_layer)
        ### Data
        self._weights: list[np.matrix] = self._initialize_weights()
        self._activation: list[np.matrix] = None
        self._gradient: list[np.matrix] = [
            np.matrix(np.zeros(theta.shape)) for theta in self.weights
        ]

    @staticmethod
    def activation_function(z: float | np.ndarray):
        return ut.sigmoid(z)

    @staticmethod
    def loss_function(
        labels: np.ndarray | np.matrix, prediction: np.ndarray | np.matrix
    ):
        return ut.cross_entropy(labels, prediction)

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

    def _initialize_weights(self) -> list[np.matrix]:
        weights = []
        for i in range(1, self.number_hidden_layers + 1):
            neurons_left = self.neurons_per_layer[i - 1] + 1  # +1 for bias
            neurons_right = self.neurons_per_layer[i]
            random_theta = self._random_matrix_xavier(neurons_right, neurons_left)
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
                0, self._random_matrix_xavier(self.neurons_per_layer[0], n + 1)
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
        return self.gradient

    def backpropagation(
        self, examples: np.matrix, labels: np.matrix
    ) -> list[np.matrix]:
        self._forward_pass(examples)
        return self._backward_pass(labels)

    def _regularized_gradient(self, reg_lambda: float) -> list[np.matrix]:
        for L in range(self.number_hidden_layers + 1):
            gradient = self.gradient[L]
            m = gradient.shape[0]  # number of examples
            regularized_theta = reg_lambda * self.weights[L] / m
            regularized_theta[:, 0] *= m / reg_lambda  # not regularizing the bias
            gradient += regularized_theta
            self._gradient[L] = np.matrix(gradient)
        return self.gradient
