def gradientCheck(X, y, theta, input_layer_size, hidden_layer_size, num_labels, Lambda=0,epsilon= 1e-4):
    grad_aprox = []
    theta_p = theta + epsilon
    theta_m = theta - epsilon
    
    for i in range(len(theta)):
        #theta Plus
        theta[i] = theta_p[i]
        
        #retorno de computeCost : cost,grad1,grad2,reg_J,grad1_reg,grad2_reg
        J1_plus = computeCost(X, y, theta, input_layer_size, hidden_layer_size, num_labels, Lambda)[0]
        
        #theta Minus
        theta[i] = theta_m[i]
        
        J1_minus= computeCost(X, y, theta, input_layer_size, hidden_layer_size, num_labels, Lambda)[0]
        
        theta[i] = theta [i] + epsilon
        
        
        grad_aux = ( J1_plus - J1_minus )/(2*epsilon)
        
        grad_aprox.append(grad_aux)
    
    
    grad_aprox = np.array(grad_aprox)
   
    custo,grad1_test,grad2_test =   computeCost(X, y, theta, input_layer_size, hidden_layer_size, num_labels, Lambda)[:3]
    
    grad_test = np.append(grad1_test.flatten(),grad2_test.flatten())
    
    #calcundo a diferenca entre a aproximação e o valor dado pelo back propagation da computeCust
    numerador = np.linalg.norm(grad_aprox - grad_test)                      
    denominador = np.linalg.norm(grad_aprox) + np.linalg.norm(grad_test)            
    diferenca =  numerador/denominador
    
    if diferenca < epsilon :
        print("O gradiente esta correto!")
    else:
        print("O gradiente esta errado!")
    
    return grad_aprox, diferenca
