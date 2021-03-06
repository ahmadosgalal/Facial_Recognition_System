
import cv2
import os
import sys
import detectFace
from fnmatch import fnmatch
import matplotlib.pyplot as plt
import numpy as np

# Load mean vector 
mean_vec = np.load('Data/vectors/meanVector.npy') 

# Load eigen vectors 
eig_vec = np.load('Data/vectors/eigenVectors.npy')    

# Load weights
weights = np.load('Data/vectors/weights.npy')    



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
            folder = '/Real-Time/%s/'%(name)
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

folder = '/Real-Time-Output/Faces/%s/'%(facename)
root = os.path.dirname(in_path) + folder 
pattern = "*.jpg"

