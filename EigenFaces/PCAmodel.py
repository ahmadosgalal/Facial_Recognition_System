import numpy as np
from sklearn.decomposition import PCA
import cv2

# Load faces vector
faces = np.load('Data/vectors/facesVector.npy')                   


n_samples, h, w = faces.shape
faceshape = (h, w)

X = faces.reshape(n_samples, h * w)

# Find the mean vector of the training set
meanVector = np.mean(X, axis=0)

# Normalize images with mean vector
normalizedImages = X - meanVector


# Compute a PCA 
# Take the first K principal components as eigenfaces such that 95% of variance is kept
pca = PCA(n_components = 0.97).fit(normalizedImages)


# Apply PCA transformation
# X_reduced = np.dot(X, pca.components_.T) 
X_reduced = pca.transform(normalizedImages)

# Get eigen vectors
eigenVectors = pca.components_

# Get eigen values
eigenValues = pca.explained_variance_

# Get number of eigen faces
numberOfEigenfaces = len(pca.components_)

# Get eigen faces
eigenFaces = pca.components_.reshape((numberOfEigenfaces, h, w))


# Get weights
#weights = eigenVectors @ (normalizedImages).T



np.save('Data/vectors/meanVector', meanVector)                 # Storing meanVector 

np.save('Data/vectors/eigenVectors', eigenVectors)				# Storing eigenVectors 

np.save('Data/vectors/eigenValues', eigenValues)				# Storing eigenValues 

np.save('Data/vectors/eigenFaces', eigenFaces)				    # Storing eigenFaces 

np.save('Data/vectors/dataProjected', X_reduced)				# Storing dataProjected 

