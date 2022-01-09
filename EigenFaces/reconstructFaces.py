import numpy as np
import matplotlib.pyplot as plt
import PCAmodel
import plot



# Load data projected 
X_proj = np.load('Data/vectors/dataProjected.npy') 

# Load faces vector
faces = np.load('Data/vectors/facesVector.npy')                   

n_samples, h, w = faces.shape

X_inv_proj = PCAmodel.pca.inverse_transform(X_proj) 

X_proj_img = np.reshape(X_inv_proj, (n_samples, h, w))



# Show first 50 reconstructed faces
face_titles = ["Face %d" % i for i in range(X_proj_img.shape[0])]
plot.plot_portraits(X_proj_img, face_titles, X_proj_img.shape[2], X_proj_img.shape[1], 10, 10) 
