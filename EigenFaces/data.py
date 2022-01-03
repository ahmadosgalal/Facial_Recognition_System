import cv2
import os
import numpy as np
#from PIL import Image
#import pickle
import inspect
import getface


# Path to images
src_file_path = inspect.getfile(lambda: None)
BASE_DIR = os.path.dirname(os.path.abspath(src_file_path))
image_dir = os.path.join(BASE_DIR, "LFW_Dataset")
#image_dir = os.path.join(BASE_DIR, "trial")



out_path = os.path.realpath("capture.py")



# Haar cascade classifier to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Navigate through images folder to get all images
for root, dirs, files in os.walk(image_dir):
    file_count = len(files)
    if(file_count > 10):
        
        # Keep track of count of images for each new face
        img_counter = 1    

        for file in files:


            if file.endswith("png") or file.endswith("jpg"):
            
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()

                img_name = "{}_{}.jpg".format(label, str(img_counter).zfill(4))  

                folder = '/real-time/Faces/%s/'%(label)
                destination_root = os.path.dirname(out_path) + folder

                                       
                # Read image
                image = cv2.imread(path)

                # Convert image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)         
                
                # Detect faces in image
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=5)

                # If there was faces detected in image
                if(len(faces) != 0):
                    # Make folder 
                    try:
                        os.makedirs(destination_root)                              
                    except:
                        pass
                
                    # For each detected face
                    for (x, y, w, h) in faces:
                        # Crop face image
                        face_img = image[y+5:y+h-5, x+5:x+w-5]             
                        # Resize to make uniform images
                        face_img = cv2.resize(face_img, (32, 32))          
                        # Save the image   
                        cv2.imwrite(destination_root + img_name, face_img)                                         
                    
                # Increment image counter
                img_counter += 1