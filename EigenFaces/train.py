import numpy as np


# Load mean vector 
mean_vec = np.load('Data/vectors/meanVector.npy') 

# Load eigen vectors 
eig_vec = np.load('Data/vectors/eigenVectors.npy')    


# Load X_train
X_train = np.load('Data/vectors/X_train.npy')                   

x_train_sample, x_h, x_w = X_train.shape

# Get weights for X_train
weights = eig_vec @ (X_train.reshape(x_train_sample, x_h * x_w) - mean_vec).T


np.save('Data/vectors/weights', weights)				    # Store weights for training data
