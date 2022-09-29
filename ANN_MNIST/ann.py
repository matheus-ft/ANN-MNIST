import numpy as np
import numpy.random as npr
from . import utils as ut


class training_hyperparams:
    def __init__(
        self,
        learning_rate: float,
        number_iterations: int,
        regularization_lambda: float = 0,
        optimizer: str = "GD",
    ) -> None:
        self.learning_rate: float = learning_rate
        self.number_iterations: int = number_iterations
        self.regularization_lambda: float = regularization_lambda
        self.optimizer: str = optimizer


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
        self._weights: list[ut.matrix] = self._initialize_weights()
        self._activation: list[ut.matrix] = None
        self._gradient: list[ut.matrix] = [
            ut.matrix(np.zeros(theta.shape)) for theta in self.weights
        ]
        self._training_data: ut.matrix = None

    @staticmethod
    def activation_function(z: float | np.ndarray):
        return ut.sigmoid(z)

    @staticmethod
    def loss_function(
        labels: np.ndarray | ut.matrix, prediction: np.ndarray | ut.matrix
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
    def weights(self) -> list[ut.matrix]:
        return self._weights

    @property
    def activation(self) -> list[ut.matrix]:
        return self._activation

    @property
    def gradient(self) -> list[ut.matrix]:
        return self._gradient

    def _neurons_list(self, n_neurons_per_layer: int) -> list[int]:
        neurons_list = [n_neurons_per_layer] * self.number_hidden_layers
        neurons_list.append(self.number_classes)
        return neurons_list

    @staticmethod
    def _random_matrix_xavier(rows: int, columns: int) -> ut.matrix:
        epislon = (6 / (rows + columns - 1)) ** 0.5
        arr = npr.rand(rows, columns)  # uniform distribution
        return ut.matrix(arr * 2 * epislon - epislon)

    def _initialize_weights(self) -> list[ut.matrix]:
        weights = []
        for i in range(1, self.number_hidden_layers + 1):
            neurons_left = self.neurons_per_layer[i - 1] + 1  # +1 for bias
            neurons_right = self.neurons_per_layer[i]
            random_theta = self._random_matrix_xavier(neurons_right, neurons_left)
            weights.append(random_theta)
        return weights

    @staticmethod
    def add_column_1s(data_matrix: ut.matrix) -> ut.matrix:
        """Add a column of 1s to left of the data given *without* modifying the
        original.

        Parameters
        ----------
        array_like : ut.matrix
            Iterable on which the column of 1s is added

        Returns
        -------
        ut.matrix
            Same iterable with the additional column of 1s to the left
        """
        m = data_matrix.shape[0]
        ones = np.ones((m, 1))
        array = np.concatenate((ones, data_matrix), axis=1)
        return ut.matrix(array)

    def _forward_pass(self, examples: ut.matrix) -> None:
        self._training_data = examples
        m = examples.shape[0]  # number of examples
        n = examples.shape[1]  # number of features
        if len(self.weights) == self.number_hidden_layers:
            theta_0 = self._random_matrix_xavier(self.neurons_per_layer[0], n + 1)
            self._weights.insert(0, theta_0)  # meaning the bias is missing
            self._gradient.insert(0, ut.matrix(np.zeros(theta_0.shape)))
        if self.activation is None:
            self._activation = [
                ut.matrix(np.zeros((m, n_l))) for n_l in self.neurons_per_layer
            ]
        ex = self.add_column_1s(examples)
        z_0 = ex @ self.weights[0].T
        self._activation[0] = self.activation_function(z_0)
        for j in range(1, self.number_hidden_layers + 1):
            a_j_1 = self.add_column_1s(self.activation[j - 1])
            z_j = a_j_1 @ self.weights[j].T
            self._activation[j] = self.activation_function(z_j)

    def _backward_pass(self, labels: ut.matrix) -> list[ut.matrix]:
        m = labels.shape[0]  # number of examples
        H = self.number_hidden_layers
        prediction = self.activation[H]  # last activation, i.e. output of the net
        delta_L = prediction - labels
        for L in range(H - 1, -1, -1):  # goes to 0 bc the training data is not included
            a_L = self.activation[L]
            a_L_1s = self.add_column_1s(a_L)
            D = delta_L.T @ a_L_1s
            self._gradient[L + 1] = D / m
            theta_L = self.weights[L + 1][:, 1:]  # discarding the bias
            delta_L = delta_L @ theta_L * a_L * (1 - a_L)
        ex = self.add_column_1s(self._training_data)
        D = delta_L.T @ ex
        self._gradient[0] = D / m
        return self.gradient

    def backpropagation(
        self, examples: ut.matrix, labels: ut.matrix
    ) -> list[ut.matrix]:
        self._forward_pass(examples)
        return self._backward_pass(labels)

    def _regularized_gradient(self, reg_lambda: float) -> list[ut.matrix]:
        for L in range(self.number_hidden_layers + 1):
            gradient = self.gradient[L]
            m = gradient.shape[0]  # number of examples
            regularized_theta = reg_lambda * self.weights[L] / m
            regularized_theta[:, 0] *= m / reg_lambda  # not regularizing the bias
            gradient += regularized_theta
            self._gradient[L] = ut.matrix(gradient)
        return self.gradient

    def _gradient_descent(
        self,
        number_iterations: int,
        examples: ut.matrix,
        labels: ut.matrix,
        reg_lambda: float,
        learning_rate: float,
    ) -> tuple[list[ut.matrix], list[float]]:
        cost_history = []
        for _ in range(number_iterations):
            self.backpropagation(examples, labels)
            gradient_list = self._regularized_gradient(reg_lambda)
            cost_history.append(self.loss_function(labels, self.activation[-1]))
            for i in range(len(gradient_list)):
                theta = self.weights[i]
                theta -= learning_rate * gradient_list[i]
                self._weights[i] = ut.matrix(theta)
        return self.weights, cost_history

    def _optimize(
        self,
        examples: ut.matrix,
        labels: ut.matrix,
        training_params: training_hyperparams,
    ) -> tuple[list[ut.matrix], list[float]]:
        learning_rate = training_params.learning_rate
        number_iterations = training_params.number_iterations
        reg_lambda = training_params.regularization_lambda
        optimizer = training_params.optimizer
        if optimizer == "GD":
            return self._gradient_descent(
                number_iterations, examples, labels, reg_lambda, learning_rate
            )
        else:
            raise Exception(f"Method {optimizer} not available")

    def fit(
        self,
        examples: ut.matrix,
        labels: ut.matrix,
        training_params: training_hyperparams,
    ) -> tuple[list[ut.matrix], list[float]]:
        return self._optimize(examples, labels, training_params)

    def predict(self, examples: ut.matrix) -> ut.matrix:
        """Predicts the classes expected for each example.

        Each row is the predicted class (numbered from 0 to `self.number_classes` - 1) for the respective example (row in the matrix given).

        Parameters
        ----------
        examples : ut.matrix
            Data matrix containing one example per row and with each column representing a feature

        Returns
        -------
        ut.matrix
            A column matrix with the predicted classes
        """
        activation: list[ut.matrix] = []
        ex = self.add_column_1s(examples)
        z_0 = ex @ self.weights[0].T
        activation.append(self.activation_function(z_0))
        for j in range(1, self.number_hidden_layers + 1):
            a_j_1 = self.add_column_1s(activation[j - 1])
            z_j = a_j_1 @ self.weights[j].T
            activation.append(self.activation_function(z_j))
        return np.argmax(activation[-1], axis=1).reshape((examples.shape[0], 1))
