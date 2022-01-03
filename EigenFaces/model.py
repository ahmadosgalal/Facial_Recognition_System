import cv2
import numpy as np
#from matplotlib import pyplot as plt
from numpy import linalg as LA
#from scipy.spatial import distance
import os
from fnmatch import fnmatch

# Get images
def images(dataset):
    if dataset == 'preset':            #For accuracy on a dataset
        folder = '/preset/'

    elif dataset == 'real-time':     #For real-time
        folder = '/real-time/'        

    in_path = os.path.realpath("train.py")             
    root = os.path.dirname(in_path) + folder           

    pattern = "*.jpg"

    images_path = []

	# Collect all images for training
    for path, subdirs, files in os.walk(root):        
        for name in files:
            if fnmatch(name, pattern):
                images_path.append(os.path.join(path, name))

    images = []
    for img_path in images_path:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale input
        for_img_size = img
        
        # Flatten image to M*N*1 vector 
        img = img.flatten() 

		# Append image to vector of images
        images.append(img) 

	# Get the size of the image
    size = for_img_size.size
      
    return images, len(images), size, images_path 



# Find the mean vector of the training set
def mean_vec(images, size, m):            
    sum_input = [0 for i in range(size)]
    for img in images:
        sum_input += img 

    mean_input = sum_input/float(m)
    return mean_input



# Normalize images with mean vector
def normalise(images, mean_vec): 
    normalized_images = []
    for img in images:
        img = img - mean_vec
        normalized_images.append(img)

    return normalized_images


# PCA Function
def pca(normalized_images, k):                    
    A = np.transpose(np.matrix(normalized_images))          # A = [img1 img2 img3 ...]
    mat_pseudo = np.matmul(A.transpose(), A) 				# M*M
    eig_val, v = LA.eig(mat_pseudo)            				# v = eigen vectors for A(trans)*A

    M = len(normalized_images)

	# To keep chosen eigen vectors
    eig_vec = []

	# To keep track of how many eigen vectors are chosen
    c = 0

	# Get best k eigen vectors
    for m in range(M):
        if c < k:
            u = np.matmul(A, v[:, m].real)  

			# Normalize ||u|| = 1      
            x = np.linalg.norm(u)            	
            u_ = u/x
            
            eig_vec.append(u_)
            c += 1
        else:
            break

	# Get weights of all the training set
    weights = []                           
    for img in normalized_images:
        w_main = []
        for u in eig_vec:
            w = np.matmul(u.transpose(), img)
            w_main.append(w)

        weights.append(np.array(w_main))

    return eig_vec, weights