


#%%
## aplicando scaling nos vetores que transformarei em imagem
line_vector = np.delete(theta1,0,1)
def scaling(line_vector):
    over  = np.max(line_vector)
    under = np.min(line_vector)
    return (line_vector - under)/(over - under)
    
line_vector=scaling(line_vector)
for i in range(len(line_vector)):
    plt.imshow(line_vector[i,:].reshape(20,20,order='F'),cmap='hot')
    plt.axis('off')
    
