import numpy as np
#import cv2
import model
import getface
import capture
import test
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt


# Visualise portraits from the dataset
def plot_eigen_portraits(images, titles, h, w, n_row, n_col):
    plt.figure(figsize=(2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
        plt.show()



# Get name of new face
#name = sys.argv[1]

# Capture photos from camera
#capture.capture(name)                    

# Get the face from the photos
#getface.getface(name)                    


images, len_images, size, images_path = model.images('real-time')        
print(size)

#mean_vec = model.mean_vec(images, size, len_images)

# Find the mean vector of the training set
mean_vec = np.mean(images, axis=0)


#normalized_images = model.normalise(images, mean_vec)

# Normalize images with mean vector
normalized_images = images - mean_vec




from sklearn.decomposition import PCA
# Compute  PCA 
#pca = PCA(n_components = 0.95)
#pca.fit(normalized_images)

# apply PCA transformation
#reduced = pca.transform(normalized_images)

# Get number of eigen faces
#number_of_eigenfaces = len(pca.components_)
#print(number_of_eigenfaces)

# Get eigen faces
#eigen_faces = pca.components_.reshape((number_of_eigenfaces, 32, 32))

#print(eigen_faces)

#eigenface_titles = ["eigenface %d" % i for i in range(len(eigen_faces))]

# Plotting first 100 of 300 faces
#plot_eigen_portraits(eigen_faces, eigenface_titles, 32, 32, 1, 10) 


#print(len(normalized_images))
#k = (len(normalized_images)/2) + 1
#print(k)
k = 194

#Use PCA to get Eigenfaces
eig_vec, weights = model.pca(normalized_images, k)                    

eigenface_titles = ["eigenface %d" % i for i in range(len(eig_vec))]

# Plotting first 100 of 300 faces
#plot_eigen_portraits(eig_vec, eigenface_titles, 32, 32, 1, 10) 
#print('\n')
#print('\n')
print(eig_vec)

#print(weights)
#print('\n')
#print('\n')

# Store mean vector for real time
np.save('real-time/vectors/mean_vec',mean_vec)              

# Store eigen vector for real time
np.save('real-time/vectors/eig_vec',eig_vec)               

# Store weights for real time
np.save('real-time/vectors/weights',weights)                


# Image paths for matching the index of nearest face class
pd.DataFrame(images_path).to_csv('real-time/vectors/image_path.csv', index = False) 

