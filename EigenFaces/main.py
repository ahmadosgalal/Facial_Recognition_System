import cv2
import numpy as np
import test
import pandas as pd
import os

# Load training mean vector 
mean_vec = np.load('real-time/vectors/mean_vec.npy') 

# Load training eigen vector 
eig_vec = np.load('real-time/vectors/eig_vec.npy')    

# Load training weights
weights = np.load('real-time/vectors/weights.npy')    

# Haar cascade classifier to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

i = 2

if(i == 1):
    # Open camera             
    cap = cv2.VideoCapture(0)

    while 1:
        # Input video stream
        ret, img = cap.read()
                                            
        # Convert image to grayscale                                    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)            

        # Detect faces in image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)        

        # For each detected face
        for (x, y, w, h) in faces:
            # Crop image to get face 
            face_img = img[y:y+h, x:x+w]                        
            face_img = face_img[:, :, 0]

            # Resize image
            face_img = cv2.resize(face_img, (32, 32))

            # Check class of image
            ind, dis = test.test_img(mean_vec, eig_vec, weights, face_img)                    

            # If there is a match
            if dis != -1:
                # Get index
                list_images = list(pd.read_csv('real-time/vectors/image_path.csv')['0']) 

                # Get name of face class
                text_name = os.path.basename(os.path.dirname(list_images[ind]))             
            
            # No match
            else: 
                text_name = ''
                
            # Draw rectangle on face 
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)                                 

            # Display name on face
            cv2.putText(img, text_name, (x, y-2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)



        cv2.imshow('Main', img)

        # Output from waitKey is logically AND with 0xFF so that last 8 bits can be accessed
        k = cv2.waitKey(30) & 0xff

        # If esc is pressed, close camera
        if k == 27:
            break
        
        
    cap.release()
    cv2.destroyAllWindows()

else:
    # Read image
    image = cv2.imread("angelina_jolie_0004.jpg")

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)         
                
    # Detect faces in image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=5)
    
    # If there were faces detected in image
    if(len(faces) != 0):
        print("Face detected")
        # For each detected face
        for (x, y, w, h) in faces:
            # Crop image to get face 
            face_img = image[y:y+h, x:x+w]                        
            face_img = face_img[:, :, 0]

            # Resize image
            face_img = cv2.resize(face_img, (32, 32))

            # Check class of image
            ind, dis = test.test_img(mean_vec, eig_vec, weights, face_img)                    

            # If there is a match
            if dis != -1:
                # Get index
                list_images = list(pd.read_csv('real-time/vectors/image_path.csv')['0']) 

                # Get name of face class
                text_name = os.path.basename(os.path.dirname(list_images[ind]))       


            # No match
            else: 
                text_name = ''

        print(text_name)
                