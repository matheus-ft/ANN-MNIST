import numpy as np
import pandas as pd


def get_data():
    X = pd.read_csv("data/imageMNIST.csv", decimal=",", header=None)
    X = np.array(X)

    y_raw = pd.read_csv("data/labelMNIST.csv", decimal=",", header=None)
    y_raw = np.array(y_raw)

    n_examples = y_raw.shape[0]
    n_classes = len(np.unique(y_raw))
    y = np.zeros((n_examples, n_classes))
    for i, j in zip(y_raw, y):
        j[i - 1] = 1

    return X, y


def split_data(X, y, train_len = 1., eval_len = .0, test_len = .0):
    data_s = {
        "X_train": X,
        "X_eval": [],
        "X_test": [],
        "y_train": y,
        "y_eval": [],
        "y_test": [],
    }
    if train_len + eval_len + test_len != 1.0:
        raise Exception("The sum of percentages is different of 100%")
    elif train_len == 0:
        raise Exception("No values to train")
    elif eval_len == 0 and test_len == 0:
        return data_s
    
    lengths = {
        "train": train_len, 
        "eval": eval_len, 
        "test": test_len,
    }
    lengths = dict(sorted(lengths.items(), key=lambda item: item[1]))
    indexes = {"train": np.array([], dtype=int), "eval": np.array([], dtype=int), "test": np.array([], dtype=int)}

    y_each, y_each_n = np.unique(y, axis=0, return_counts=True)
    y_each_i = [np.random.permutation(np.where((y==each).all(-1))[0]) for each in y_each]
    initial_len = [0]*len(y_each)
    middle_len = [0]*len(y_each)

    for i, group in enumerate(lengths):
        for n in range(len(y_each)):
            if i == 0:
                initial_len[n] = int(len(y_each_i[n]) * lengths[group])
                indexes[group] = np.append(indexes[group], [y_each_i[n][:initial_len[n]]])
            elif i == 1:
                middle_len[n] = initial_len[n] + int(len(y_each_i[n]) * lengths[group])
                indexes[group] = np.append(indexes[group], [y_each_i[n][initial_len[n]:middle_len[n]]])
            else:
                indexes[group] = np.append(indexes[group], [y_each_i[n][middle_len[n]:]])

    for group in lengths:
        data_s[f"X_{group}"] = X[indexes[group]]
        data_s[f"y_{group}"] = y[indexes[group]]

    return data_s

