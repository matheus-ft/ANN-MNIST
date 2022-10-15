# %%
import pandas as pd
import numpy as np
from ANN_MNIST import assembly_nn, gradientDescent, prediction
from datetime import datetime as dt
import matplotlib.pyplot as plt


# %%
X = pd.read_csv("data/imageMNIST.csv", decimal=",", header=None)
X = np.array(X)
X, X.shape

# %%
y_raw = pd.read_csv("data/labelMNIST.csv", decimal=",", header=None)
y_raw = np.array(y_raw)
y_raw, y_raw.shape

# %%
n_examples = y_raw.shape[0]
n_classes = len(np.unique(y_raw))
y = np.zeros((n_examples, n_classes))
for i, j in zip(y_raw, y):
    j[i - 1] = 1
y, y.shape

# %%
n_features = X.shape[1]
hidden_layers = [25]
nn = assembly_nn(n_features, n_classes, hidden_layers)

# %%
now = dt.now()
learning_rate = 0.8
iterations = 800
reg_lambda = 1
nn, J_hist = gradientDescent(X, y, nn, learning_rate, iterations, reg_lambda)
time_elapsed = dt.now() - now

# %%
print(f"Time took for training: {time_elapsed}")
plt.plot(range(iterations), J_hist)

# %%
pred = prediction(X, nn)
accuracy = sum(pred[:, np.newaxis] == y_raw)[0] / 5000 * 100
print(f"Training set accuracy: {accuracy} %")

# %%
