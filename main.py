from datetime import datetime as dt
import numpy as np
from scipy.io import loadmat
from neural_network import (
    randInitializeWeights, 
    gradientDescent, 
    prediction,
)


agora = dt.now()

def get_data():
    mat = loadmat("ex3data1.mat")

    X = mat["X"]
    y_t = mat["y"]

    y_t = y_t.reshape(y_t.shape[0],)

    y = np.zeros((y_t.shape[0], len(np.unique(y_t))))

    for i, j in zip(y_t, y):
        j[i-1] = 1.
    
    return X, y

mat = loadmat("ex3data1.mat")

X = mat["X"]
y = mat["y"]

def assembly_nn(input_layer: int,hidden_layer: list,output_layer: int):
    theta = []
    theta

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

initial_theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size,num_labels)
initial_theta = np.append(initial_theta1.flatten(),initial_theta2.flatten())


theta,J_history = gradientDescent(X,y,initial_theta,0.8,50,1,input_layer_size,hidden_layer_size,num_labels)
theta1 = theta[:((input_layer_size+1)*hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
theta2 = theta[((input_layer_size+1)*hidden_layer_size):].reshape(num_labels,hidden_layer_size+1)

print("tempo:",dt.now() - agora)

pred = prediction(X,theta1,theta2)
print(pred.shape)
print(pred[:3])
print("Training Set Accuracy:",sum(pred[:,np.newaxis]==y)[0]/5000*100,"%")
