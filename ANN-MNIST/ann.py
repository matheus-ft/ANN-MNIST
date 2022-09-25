import numpy as np
import numpy.random as npr
import utils as ut


class ANN:
    def __init__(
        self,
        n_classes: int,
        n_neurons_per_layer: int,
        n_hidden_layers: int = 1,
        activation_function=ut.sigmoid,
        loss_function=ut.cross_entropy,
    ) -> None:
        ### Architecture
        self.__number_classes = n_classes
        self.__number_hidden_layers = n_hidden_layers
        self.__neurons_per_layer: list[int] = self._neurons_list(n_neurons_per_layer)
        self.__activation_function = activation_function
        self.__loss_function = loss_function
        ### Data
        self._weights: list[np.matrix] = self._initialize_weights()
        self._activation: list[np.matrix] = None

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
    def loss_function(self):
        return self.__loss_function

    @property
    def weights(self) -> list[np.matrix]:
        return self._weights

    @property
    def activation(self) -> list[np.matrix]:
        return self._activation

    def _neurons_list(self, n_neurons_per_layer: int) -> list[int]:
        neurons_list = [n_neurons_per_layer] * self.number_hidden_layers
        neurons_list.append(self.number_classes)
        return neurons_list

    def _initialize_weights(self) -> list[np.matrix]:
        weights = []
        for i in range(self.number_hidden_layers + 1):
            neurons_left = self.neurons_per_layer[i]
            neurons_right = self.neurons_per_layer[i + 1]
            epislon = (6 / (neurons_left + neurons_right)) ** 0.5
            random_theta = npr.rand(neurons_right, neurons_left + 1)  # +1 due to bias
            weights.append(random_theta * 2 * epislon - epislon)
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

    def _forward_pass(self, examples: np.matrix) -> np.matrix:
        m = examples.shape[0]  # number of examples
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
        return self.activation[-1]

    def _back_propagation(self):
        pass

    def fit(self, examples, labels):
        pass

    def predict(self, new_examples):
        pass
