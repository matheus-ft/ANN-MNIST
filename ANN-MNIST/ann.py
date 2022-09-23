import numpy as np
import numpy.random as npr
from scipy.special import expit as sigmoid  # faster than using np.exp


class ANN:
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_hidden_neurons: int,
        n_hidden_layers: int = 1,
    ) -> None:
        self.__number_features = n_features
        self.__number_classes = n_classes
        self.__number_hidden_layers = n_hidden_layers
        self.__number_neurons: list[int] = self._neurons_list(n_hidden_neurons)
        self._weights: list[np.ndarray] = self._initialize_weights()

    @property
    def number_features(self) -> int:
        return self.__number_features

    @property
    def number_classes(self) -> int:
        return self.__number_classes

    @property
    def number_hidden_layers(self) -> int:
        return self.__number_hidden_layers

    @property
    def number_neurons(self) -> list[int]:
        return self.__number_neurons

    @property
    def weights(self) -> list[np.ndarray]:
        return self._weights

    def _neurons_list(self, n_hidden_neurons: int) -> list[int]:
        neurons_list = [self.number_features]
        neurons_list.extend([n_hidden_neurons] * self.number_hidden_layers)
        neurons_list.append(self.number_classes)
        return neurons_list

    def _initialize_weights(self) -> list[np.ndarray]:
        weights = []
        for i in range(self.number_hidden_layers + 1):
            neurons_left = self.number_neurons[i]
            neurons_right = self.number_neurons[i + 1]
            epislon = (6 / (neurons_left + neurons_right)) ** 0.5
            random_theta = npr.rand(neurons_left, neurons_right + 1)  # +1 due to bias
            weights.append(random_theta * 2 * epislon - epislon)
        return weights

    def _forward_pass(self):
        pass

    def _back_propagation(self):
        pass

    def fit(examples: np.ndarray, labels: np.ndarray) -> np.ndarray:
        pass

    def predict(new_examples: np.ndarray) -> np.ndarray:
        pass


def sigmoid_prime(z: float) -> float:
    a = sigmoid(z)
    return a * (1 - a)
