# %%
from ANN_MNIST import (
    metrics,
    assembly_nn,
    gradientDescent,
    prediction,
    gradient_check,
    theta_meaning,
    get_data,
    split_data,
)
import numpy.random as npr
import numpy as np
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
X, y = get_data()


# %%
data_s = split_data(X, y, train_len=.6, eval_len=.2, test_len=.2)
n_features = data_s["X_train"].shape[1]
n_classes = data_s["y_train"].shape[1]
hidden_layers = [25]
learning_rate = 0.8
iterations = 100
lambda_auto = [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10]
lambda_selected = {"lambda": -1.0, "error": -1.0}
J_eval_x_lambda = []
for lambda_x in lambda_auto:
    nn = assembly_nn(n_features, n_classes, hidden_layers)
    nn, J_hist, J_eval_hist = gradientDescent(data_s["X_train"], data_s["y_train"], nn, learning_rate, iterations, lambda_x, data_s["X_eval"], data_s["y_eval"])
    J_eval_x_lambda.append([lambda_x, min(J_eval_hist)])
    if lambda_selected["error"] < 0 or min(J_eval_hist) < lambda_selected["error"]:
        lambda_selected["lambda"] = lambda_x
        lambda_selected["error"] = min(J_eval_hist)

# %%
data_s = split_data(X, y, train_len=.6, eval_len=.2, test_len=.2)
n_classes = data_s["y_train"].shape[1]
n_features = data_s["X_train"].shape[1]
hidden_layers = [25]
nn = assembly_nn(n_features, n_classes, hidden_layers)
learning_rate = 0.8
iterations = 2000
reg_lambda = lambda_selected["lambda"]

# %%
now = dt.now()
nn, J_hist, J_eval_hist = gradientDescent(data_s["X_train"], data_s["y_train"], nn, learning_rate, iterations, reg_lambda, data_s["X_eval"], data_s["y_eval"])
time_elapsed = dt.now() - now

# %%
print(f"Time took for training: {time_elapsed}")
plt.plot(range(iterations), J_hist)
plt.plot(range(iterations), J_eval_hist)

# %%
y_pred = prediction(data_s["X_test"], nn)[:, np.newaxis]
y_real = (np.argmax(data_s["y_test"],axis=1)+1).reshape(data_s["y_test"].shape[0], 1)
accuracy = sum(y_pred == y_real)[0] / y_real.shape[0] * 100
print(f"Accuracy: {accuracy} %")
test_loss = metrics(data_s["X_test"], data_s["y_test"], nn, reg_lambda)[-1]
print(f"Test loss: {test_loss}")

# %%
theta_meaning(nn[0], color_map="hot")

# %%
