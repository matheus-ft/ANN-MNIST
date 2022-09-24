import numpy as np
import numpy.random as npr
from scipy.special import expit as sigmoid  # faster than using np.exp
import utils as ut


class ANN:
    def __init__(
        self,
        n_classes: int,
        n_neurons_per_layer: int,
        n_hidden_layers: int = 1,
        activation_function=sigmoid,
    ) -> None:
        ### Architecture
        self.__activation_function = sigmoid
        self.__number_classes = n_classes
        self.__number_hidden_layers = n_hidden_layers
        self.__neurons_per_layer: list[int] = self._neurons_list(n_neurons_per_layer)
        ### Data
        self._weights: list[np.matrix] = self._initialize_weights()
        self._activation: list[ut.vector] = None

    @property
    def activation_function(self):
        return self.__activation_function

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
    def activation(self) -> list[ut.vector]:
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
    def add_column_1s(array_like: np.matrix | ut.vector) -> np.matrix | ut.vector:
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
        m = array_like.shape[0]  # rows
        n = array_like.shape[1]  # columns
        ones = np.ones((m, 1))
        array = np.concatenate((ones, array_like), axis=1)
        return ut.vector(array) if n == 1 else np.matrix(array)

    def _forward_pass(self, example: ut.vector) -> ut.vector:
        """Calculates the activation for every neuron on the network.

        Parameters
        ----------
        example : ut.vector
            A single training example as a vector

        Returns
        -------
        ut.vector
            The activation of the network; its number of rows is the same as the number of classes
        """
        if self.activation is None:
            self._activation = [
                ut.vector(np.zeros(n_l)) for n_l in self.neurons_per_layer
            ]
        ex = self.add_column_1s(example)
        z_0 = self.weights[0] @ ex
        self._activation[0] = sigmoid(z_0)
        for j in range(1, len(self.neurons_per_layer)):
            a_j_1 = self.add_column_1s(self.activation[j - 1])
            z_j = self.weights[j] @ a_j_1
            self._activation[j] = ut.vector(self.activation_function(z_j))
        return self.activation[-1]
