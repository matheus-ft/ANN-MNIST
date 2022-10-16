import numpy as np
import matplotlib.pyplot as plt


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


def _metrics(X,y,nn,Lambda):
    m = X.shape[0]
    X = np.hstack((np.ones((m,1)),X))

    a = _acrossNN(X, nn)

    cost = sum(sum([-y_col*np.log(a_col)-(1-y_col)*np.log(1-a_col) for y_col, a_col in zip(y.T, a[-1].T)])) / m

    reg_J = cost + Lambda/(2*m)*(sum([np.sum(layer[:,1:]**2) for layer in nn]))

    return m, X, a, cost, reg_J


def _computeCost(X,y,nn, Lambda):
    m, X, a, cost, reg_J = _metrics(X,y,nn,Lambda)

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


def gradientDescent(X_train,y_train,nn,alpha,nbr_iter,Lambda,X_eval=[],y_eval=[],printe=True):
    J_history = []
    J_history_eval = []
    reg_J_eval = 0

    for i in range(nbr_iter):
        if X_eval != []:
            reg_J_eval = _metrics(X_eval,y_eval,nn,Lambda)[-1]
            J_history_eval.append(reg_J_eval)
        cost,reg_J,grads,grads_reg = _computeCost(X_train,y_train,nn,Lambda)
        nn = [layer - alpha * grad_reg for layer, grad_reg in zip(nn, grads_reg)]
        J_history.append(reg_J)
        if printe and i % 10 == 0:
            print(f"train:{reg_J}, eval:{reg_J_eval}")

    return nn,J_history,J_history_eval


def prediction(X,nn):
    m = X.shape[0]
    X = np.hstack((np.ones((m,1)),X))

    a = _acrossNN(X, nn)

    return np.argmax(a[-1],axis=1)+1


def gradient_check(X, y, nn, epsilon=1e-4):
    backprop_grads = _computeCost(X, y, nn, 0)[2]
    error = []
    for k in range(len(nn)):
        layer = nn[k]
        m, n = layer.shape
        for i in range(m):
            for j in range(n):
                og_theta_ij = layer[i, j]

                theta_plus = og_theta_ij + epsilon
                nn[k][i, j] = theta_plus
                J_plus = _computeCost(X, y, nn, 0)[0]

                theta_minus = og_theta_ij - epsilon
                nn[k][i, j] = theta_minus
                J_minus = _computeCost(X, y, nn, 0)[0]

                nn[k][i, j] = og_theta_ij
                aprox_grad_ij = (J_plus - J_minus) / (2 * epsilon)
                backprop_grad_ij = backprop_grads[k][i, j]
                error.append(abs(aprox_grad_ij - backprop_grad_ij))
    error = np.mean(error)
    return True if error < epsilon else False

def scaling(line_vector):
    over = np.max(line_vector)
    under = np.min(line_vector)
    return (line_vector - under) / (over - under)


def theta_meaning(theta, color_map=None):
    theta = theta[:, 1:]  # discarding the bias
    hidden_nodes = theta.shape[0]
    images = [scaling(theta[i]).reshape(20, 20, order="F") for i in range(hidden_nodes)]
    grid_size = np.sqrt(hidden_nodes)
    nrows = int(grid_size) if grid_size.is_integer() else int(grid_size) + 1
    ncols = int(grid_size) if grid_size.is_integer() else int(grid_size) - 1
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    _, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    axes = axes.flatten()[: len(imgs)]
    for img, ax in zip(imgs, axes.flatten()):
        ax.imshow(img, cmap=color_map)
        ax.axis("off")
