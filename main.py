from datetime import datetime as dt
import numpy as np
from scipy.io import loadmat
from neural_network import (
    assembly_nn, 
    gradientDescent, 
    prediction,
)


agora = dt.now()

def get_data():
    mat = loadmat("ex3data1.mat")

    X = mat["X"]
    Y = mat["y"]

    y_t = Y.reshape(Y.shape[0],)

    y = np.zeros((y_t.shape[0], len(np.unique(y_t))))

    for i, j in zip(y_t, y):
        j[i-1] = 1.
    
    return X, y, Y


input_layer_size = 400
hidden_layers_size = [25]
num_labels = 10

X, y, y_t = get_data()

nn = assembly_nn(input_layer_size, num_labels, hidden_layers_size)

nn,J_history = gradientDescent(X,y,nn,0.8,50,1)

print("tempo:",dt.now() - agora)

pred = prediction(X,nn)
print("Training Set Accuracy:",sum(pred[:,np.newaxis]==y_t)[0]/5000*100,"%")
