# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 14:15:28 2022

@author: othav
"""


# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# %%
mat = loadmat("ex3data1.mat")
X = mat["X"]
y = mat["y"]

# %%
#fig,axis = plt.subplots(10,10,figsize=(8,8))
#for i in range(10):
 #   for j in range(10):
  #      axis[i,j].imshow(X[np.random.randint(0,5001),:].reshape(20,20,order="F"),cmap="hot")
   #     axis[i,j].axis("off")


# %%
def sigmoid(z):
    return 1/(1+np.exp(-z))


# %%
def sigmoidGradient(z):
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid*(1-sigmoid)


# %%
def computeCost(X,y,theta,layer_list, Lambda):
    theta_list = [] # lista de thetas
    ponteiro = 0 # ponteiro para indicar qual parte da lista estamos.
    
    #Construindo as matrizes theta1, theta2.... thetan
    for i in range(len(layer_list)-1):
        
        theta_tmp = theta[ponteiro : ponteiro + ((layer_list[i]+1)*layer_list[i+1])].reshape(layer_list[i+1],layer_list[i]+1)
        
        theta_list.append(theta_tmp)
        
        ponteiro += (layer_list[i]+1)*layer_list[i+1] # setando o ponteiro
        
    

  
    
    m = X.shape[0]
    J = 0
    X = np.hstack((np.ones((m,1)),X))
    y10 = np.zeros((m,num_labels))
    
    a = [] # lista de ativações
    a.append( sigmoid(X @ theta_list[0].T) ) # adicionando o a0, ou seja features * first_hidden_layer
    
    # X (5000,401) * (25,401)  logo a[0] = (5000 , 25) e  a[0][0]= (25,)
    
    #adicionando o Bias ao a0 antes de realizar os calculos
    a[0] = np.hstack((np.ones((m,1)),a[0]))
    
    #Calculando a1,a2...an-1 e salvando em uma lista
    for i in range(1,len(theta_list)-1):
        a_tmp = sigmoid( a[i-1] @ theta_list[i].T )
        a_tmp = np.hstack((np.ones((m,1)),a_tmp))
        a.append(a_tmp)
        
    # para a ultima camada
    a.append(sigmoid(a[-1] @ theta_list[-1].T))
    
    a1 = a[0]
    a2 = a[1]
        
   # a1 = sigmoid(X @ theta1.T)
   # a1 = np.hstack((np.ones((m,1)),a1))
   # a2 = sigmoid(a1 @ theta2.T)
    
    for i in range(1,num_labels+1):
        y10[:,i-1][:,np.newaxis] = np.where(y==i,1,0)
    for j in range(num_labels):
        J = J + sum(-y10[:,j]*np.log(a[-1][:,j])-(1-y10[:,j])*np.log(1-a[-1][:,j]))
        
    cost = J/m
    
    reg_J = 0
    for t in theta_list:
        reg_J += np.sum(t[:,1:]**2)
    
    #reg_J = cost + Lambda/(2*m)*(np.sum(theta1[:,1:]**2)+np.sum(theta2[:,1:]**2))
    
    reg_J = cost + reg_J*Lambda/(2*m) 
         
    grad_list = [] # lista de gradientes
    
    #Criando uma lista contendo grad1,grad2...gradn, onde todos os elementos são iguais a 0
    for i in range(len(theta_list)):
        grad_list.append(np.zeros((theta_list[i].shape)))
                        
    grad1 = grad_list[0]
    grad2 = grad_list[1]
                                 
    for i in range(m):
        xi = X[i,:]
        #a é a lista com as ativações, i é o exemplo, e j é a qual das camadas estamos se referindo
        aji = list(range(len(a)))  # nao importa o valor desta lista, então eu a criei como [1,2,...,n]
        dj  = aji  # nao importa o valor desta lista, então eu a criei como = aji
        for j in range(len(a)):
            #notase que a = [a1,a2...an], e neste caso a1 = 5000 x 25, logo a[0]=a1, e a[0][i,:] são os 25 elemtnos, da i-esima linha da matriz a1
            aji[j] = (a[j][i,:])
        
        a1i = aji[0]  #lembrando que neste caso j = 0, é a camada 1
        a2i = aji[1]  # e j = 1 é a camada 2
        
        
        
        # Pela formula, O erro do ultimo nodulo é " valor_previsto  - valor_real "
        dj[-1] = aji[-1] - y10[i,:]  
        
        # Aplicando o back propagation
        # ** Aqui é -2 porque o len conta 1,2,3...,enquanto uma lista tem indice 0,1,2...
        for k in range(len(dj) -2 ,-1,-1):
            if k == 0:
                dj[k] = (theta_list[k+1].T)[1:] @ dj[k+1] * ( sigmoidGradient(np.hstack((1,xi @ theta_list[k].T)))[1:] )
            #O erro da camada n, será Theta(n+1) @ erro da camada(n+1), * ativação da camada n
            
            elif k == len(dj) -2:
                dj[k] = (theta_list[k+1].T)[1:] @ dj[k+1] * ( ((aji[k] * (1 - aji[k])))[1:] )
            
            else:
                dj[k] = (theta_list[k+1].T)[1:] @ dj[k+1] * ( ((aji[k] * (1 - aji[k])))[1:] )
        
        d2 = dj[1]
        d1 = dj[0]
        
        for  k  in range(len(grad_list)-1):
            if k == 0:
                grad_list[k] = grad_list[k] + dj[k][:,np.newaxis]@xi[:,np.newaxis].T
            else:
                grad_list[k] = grad_list[k] + dj[k][:,np.newaxis]@ (a[k-1][i,:])[:,np.newaxis].T
        
        grad_list[-1] = grad_list[-1] + dj[-1][:,np.newaxis] @ (a[-2][i,:])[:,np.newaxis].T
        
        # grad1 = grad1 + d1[1:][:,np.newaxis]@xi[:,np.newaxis].T
        # grad2 = grad2 + d2.T[:,np.newaxis]@a1i[:,np.newaxis].T    
                     
    
    for _ in range(len(grad_list)):
        grad_list[_] =  grad_list[_] * 1/m
    
    grad1 = grad_list[0]
    grad2 = grad_list[-1]
    
    grad_list_reg = []
    
    for r in range(len(grad_list)):
        grad_reg =  grad_list[r] + (Lambda/m) * np.hstack((np.zeros((theta_list[r].shape[0],1)),theta_list[r][:,1:]))
        grad_list_reg.append(grad_reg)
        #grad1_reg = grad1 + (Lambda/m)*np.hstack((np.zeros((theta1.shape[0],1)),theta1[:,1:]))
        #grad2_reg = grad2 + (Lambda/m)*np.hstack((np.zeros((theta2.shape[0],1)),theta2[:,1:]))                
    grad1_reg = grad_list_reg[0]
    grad2_reg = grad_list_reg[1]
    
    return cost, grad_list, reg_J, grad_list_reg,theta_list
    #return cost, grad1,grad2 ,reg_J, grad1_reg,grad2_reg,theta_list



# %%
def randInitializeWeights(L_in,L_out):
    epi = (6**(1/2))/(L_in+L_out)**(1/2)
    W = np.random.rand(L_out,L_in+1)*(2*epi)-epi
    return W


# %%
def gradientDescent(X,y,theta,alpha,nbr_iter,Lambda,layer_list):
    theta_list = [] # lista de thetas
    ponteiro = 0 # ponteiro para indicar qual parte da lista estamos.
    
    #Construindo as matrizes theta1, theta2.... thetan
    for i in range(len(layer_list)-1):
        
        theta_tmp = theta[ponteiro : ponteiro + ((layer_list[i]+1)*layer_list[i+1])].reshape(layer_list[i+1],layer_list[i]+1)
        
        theta_list.append(theta_tmp)
        
        ponteiro += (layer_list[i]+1)*layer_list[i+1] # setando o ponteiro
    
   # theta1 = theta[:((input_layer_size+1)*hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
   # theta2 = theta[((input_layer_size+1)*hidden_layer_size):].reshape(num_labels,hidden_layer_size+1)
    
    m = len(y)
    J_history = []
    
    for i in range(nbr_iter):
        theta = np.append(theta_list[0].flatten(),theta_list[1].flatten())
        for _ in range(2,len(theta_list)):
            theta = np.append(theta, theta_list[_].flatten())
            
        cost,grad_list_reg = computeCost(X,y,theta, layer_list ,Lambda)[2:4]
        
        for j in range(len(theta_list)):
            theta_list[j] = theta_list[j] - (alpha*grad_list_reg[j])
        
        J_history.append(cost)
        
    nn_paramsFinal = np.append(theta_list[0].flatten(),theta_list[1].flatten())
    for _ in range(2,len(theta_list)):
        nn_paramsFinal = np.append(theta, theta_list[_].flatten())

    return nn_paramsFinal,J_history,theta_list


# %%
def prediction(X,theta_list):
    m = X.shape[0]
    X = np.hstack((np.ones((m,1)),X))
    
    a = [] # lista de ativações
    a.append( sigmoid(X @ theta_list[0].T) ) # adicionando o a0, ou seja features * first_hidden_layer
    
    # X (5000,401) * (25,401)  logo a[0] = (5000 , 25) e  a[0][0]= (25,)
    
    #adicionando o Bias ao a0 antes de realizar os calculos
    a[0] = np.hstack((np.ones((m,1)),a[0]))
    
    #Calculando a1,a2...an-1 e salvando em uma lista
    for i in range(1,len(theta_list)-1):
        a_tmp = sigmoid( a[i-1] @ theta_list[i].T )
        a_tmp = np.hstack((np.ones((m,1)),a_tmp))
        a.append(a_tmp)
        
    # para a ultima camada
    a.append(sigmoid(a[-1] @ theta_list[-1].T))
    
    return np.argmax(a[-1],axis=1)+1


# %%
input_layer_size = 400
hidden_layer_size = 12
num_labels = 10
layer_list = [input_layer_size,8,hidden_layer_size,num_labels]


initial_theta1 = randInitializeWeights(input_layer_size,8)
initial_theta2 = randInitializeWeights(8,hidden_layer_size)
initial_theta3 = randInitializeWeights(hidden_layer_size,num_labels)
initial_theta = np.append(initial_theta1.flatten(),initial_theta2.flatten())
initial_theta = np.append(initial_theta,initial_theta3.flatten())
# %%

pesos_finais,J_history,theta_list = gradientDescent(X,y,initial_theta,0.8,200,1,layer_list)
#theta1 = theta[:((input_layer_size+1)*hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
#theta2 = theta[((input_layer_size+1)*hidden_layer_size):].reshape(num_labels,hidden_layer_size+1)

# %%
pred = prediction(X,theta_list)
print("Training Set Accuracy:",sum(pred[:,np.newaxis]==y)[0]/5000*100,"%")

 # %%
plt.plot(range(len(J_history)), J_history)


