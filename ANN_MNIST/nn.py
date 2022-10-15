import numpy as np


def _sigmoid(z):
    return 1/(1+np.exp(-z))


def _sigmoidGradient(z):
    sigmoid = _sigmoid(z)
    return sigmoid*(1-sigmoid)


def _acrossNN(X, nn):
    m = X.shape[0]
    a = [_sigmoid(X @ nn[0].T)]
    for layer in nn[1:]:
        a[-1] = np.hstack((np.ones((m,1)),a[-1]))
        a.append(_sigmoid(a[-1] @ layer.T))
    
    return a


def _computeCost(X,y,nn, Lambda):
    m = X.shape[0]
    X = np.hstack((np.ones((m,1)),X))
    
    a = _acrossNN(X, nn)
    
    J = sum(sum([-y_col*np.log(a_col)-(1-y_col)*np.log(1-a_col) for y_col, a_col in zip(y.T, a[-1].T)]))
        
    cost = J/m
    reg_J = cost + Lambda/(2*m)*(sum([np.sum(layer[:,1:]**2) for layer in nn]))
    
    grads = [np.zeros((layer.shape)) for layer in nn]

    for i in range(m):
        ai = [X[i,:]] + [ax[i,:] for ax in a]
        di = [ai[-1] - y[i,:]]

        for j in range(len(nn)-1, 0, -1):
            di.append((nn[j].T @ di[-1].T * _sigmoidGradient(np.hstack((1,ai[j-1] @ nn[j-1].T))))[1:])
        di.reverse()
        
        for j in range(len(grads)):
            grads[j] = grads[j] + di[j].T[:,np.newaxis] @ ai[j][:,np.newaxis].T

    grads = [grad/m for grad in grads]
    grads_reg = [grad + (Lambda/m)*np.hstack((np.zeros((layer.shape[0],1)),layer[:,1:])) for grad, layer in zip(grads,nn)]
                                 
    return cost,reg_J,grads,grads_reg


def _randInitializeWeights(L_in,L_out):
    epi = (6**(1/2))/(L_in+L_out)**(1/2)
    W = np.random.rand(L_out,L_in+1)*(2*epi)-epi
    return W


def assembly_nn(input_layer: int,output_layer: int,hidden_layer: list = []):
    layers = [input_layer] + hidden_layer + [output_layer]
    return [_randInitializeWeights(in_,out_) for in_, out_ in zip(layers[:-1], layers[1:])]


def gradientDescent(X,y,nn,alpha,nbr_iter,Lambda):
    J_history = []
    
    for i in range(nbr_iter):
        cost,reg_J,grads,grads_reg = _computeCost(X,y,nn,Lambda)
        nn = [layer - alpha * grad_reg for layer, grad_reg in zip(nn, grads_reg)]
        J_history.append(reg_J)
        if i % 10 == 0:
            print(reg_J)
    
    return nn,J_history


def prediction(X,nn):
    m = X.shape[0]
    X = np.hstack((np.ones((m,1)),X))
    
    a = _acrossNN(X, nn)
    
    return np.argmax(a[-1],axis=1)+1

