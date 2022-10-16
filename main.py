# %%
from ANN_MNIST import (
    assembly_nn,
    gradientDescent,
    prediction,
    gradient_check,
    theta_meaning,
)
import numpy.random as npr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt


# %%
random_X = npr.rand(3, 3)
random_y = np.identity(3)
test_nn = assembly_nn(3, 3, [5])
check = gradient_check(random_X, random_y, test_nn, epsilon=1e-4)
tighter_check = gradient_check(random_X, random_y, test_nn, epsilon=1e-7)
did_pass = "Gradient is correct." if check is True else "Gradient is incorrect."
did_pass_tighter = (
    "Gradient really is correct!"
    if tighter_check is True
    else "Gradient might be incorrect!"
)
print(did_pass)
print(did_pass_tighter)

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
learning_rate = 0.8
iterations = 800
reg_lambda = 1

# %%
now = dt.now()
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
theta_meaning(nn[0], color_map="hot")

# %%
