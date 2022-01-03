#import cv2
import numpy
#from matplotlib import pyplot as plt
#from numpy import linalg as LA
from scipy.spatial import distance
#import os
#from fnmatch import fnmatch


# Process unknown face for face class
def test_img(mean_vec, eig_vec, weights, input_img): 
    input_img = input_img.flatten()
    mean_input = input_img - mean_vec

    # Weights of the unknown image projected on eigenfaces
    w_main = []                                          
    for u in eig_vec:
        w = numpy.matmul(u.transpose(), mean_input)
        w_main.append(w)

    # List of distances from weights of training set and unknown face    
    dist = []                                        
    for in_w in weights:
        ed = distance.euclidean(w_main, in_w)
        dist.append(ed)

    # Argument index with minimum distance
    min_dist_index = numpy.argmin(dist) 
    min_dist = min(dist)                

    # Threshold for a face class
    if min_dist > 1000:                
        min_dist = -1

    return min_dist_index, min_dist
