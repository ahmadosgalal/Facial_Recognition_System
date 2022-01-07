import cv2
import os
from fnmatch import fnmatch


# Haar cascade classifier to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect(facename):

    # Set path
    in_path = os.path.realpath("detect.py")

    folder = '/Data/Faces/%s/'%(facename)
    root = os.path.dirname(in_path) + folder 

    pattern = "*.jpg"

    # For all images in the 'facename' directory 
    for path, subdirs, files in os.walk(root):  
        for name in files:
            if fnmatch(name, pattern):
                img_root = root + name
                img = cv2.imread(img_root)

                #Convert image color to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5) 

                # If there were faces detected in image
                if(len(faces) != 0):
                    # For each detected face
                    for (x, y, w, h) in faces:

                        # Crop face image
                        face_img = img[y+5:y+h-5, x+5:x+w-5]            
                    
                        # Resize to make uniform images
                        face_img = cv2.resize(face_img,(62, 47))         

                        # Save the image back
                        cv2.imwrite(img_root, face_img) 

                else:
                    os.remove(img_root)                


