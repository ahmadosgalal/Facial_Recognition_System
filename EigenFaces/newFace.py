
import cv2
import os
import sys
import detectFace
from fnmatch import fnmatch
import matplotlib.pyplot as plt
import numpy as np

# Load mean vector 
mean_vec = np.load('real-time/vectors/meanVector.npy') 

# Load eigen vectors 
eig_vec = np.load('real-time/vectors/eigenVectors.npy')    

# Load weights
weights = np.load('real-time/vectors/weights.npy')    



# To capture photos from webcam for a new face
def capture(name):    

    # Open camera for video capturing              
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("New_Face")

    # Keep track of count of images captured
    img_counter = 1

    while True:
        ret, frame = cam.read()

        cv2.imshow("New_Face", frame)

        # Frame is not available
        if not ret:
            break
        k = cv2.waitKey(1)

        # Esc is pressed for closing camera
        if k%256 == 27:
            print ("Closing Camera...\n")
            break

        # Space is pressed for taking photo
        elif k%256 == 32:

            img_name = "{}_{}.jpg".format(name, str(img_counter).zfill(4))  

            in_path = os.path.realpath("main.py")
            folder = '/real-time/Faces/%s/'%(name)
            root = os.path.dirname(in_path) + folder

            # Make folder for the new face
            try:
                os.makedirs(root)                              
                cv2.imwrite(root + img_name, frame)
            except:
                cv2.imwrite(root + img_name, frame)

            # Increment image counter
            print ("Photo Clicked - %d\n" %(img_counter)) 
            img_counter += 1


    cam.release()

    cv2.destroyAllWindows()


    
# Get name of new face
facename = sys.argv[1]

# Capture photos from camera
capture(facename)                    


# Detecting faces from captured photos
detectFace.detect(facename)


# Set path
in_path = os.path.realpath("newFace.py")

folder = '/real-time/Faces/%s/'%(facename)
root = os.path.dirname(in_path) + folder 
pattern = "*.jpg"


# For all images in the 'facename' directory 
for path, subdirs, files in os.walk(root):  
    for name in files:
        if fnmatch(name, pattern):
            img_root = root + name
            img = cv2.imread(img_root)
            # Convert image to grayscale
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         
                

            # Visualize
            fig, axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(8,6))
            axes.imshow(gray_image, cmap="gray")
            plt.show()

            gray_image = gray_image.reshape(1,-1)


            gray_image_weight = eig_vec @ (gray_image - mean_vec).T
            euclidean_distance = np.linalg.norm(weights - gray_image_weight, axis=0)
            best_match = np.argmin(euclidean_distance)
            top_match = np.argsort(euclidean_distance)[:7]
            print(best_match)