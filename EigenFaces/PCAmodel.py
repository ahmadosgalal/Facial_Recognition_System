import numpy as np
from sklearn.decomposition import PCA
import cv2

# Load faces vector
faces = np.load('real-time/vectors/facesVector.npy')                   


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



np.save('real-time/vectors/meanVector', meanVector)                 # Storing meanVector 

np.save('real-time/vectors/eigenVectors', eigenVectors)				# Storing eigenVectors 

np.save('real-time/vectors/eigenValues', eigenValues)				# Storing eigenValues 

np.save('real-time/vectors/eigenFaces', eigenFaces)				    # Storing eigenFaces 

np.save('real-time/vectors/dataProjected', X_reduced)				# Storing dataProjected 

