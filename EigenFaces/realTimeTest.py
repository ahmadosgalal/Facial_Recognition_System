import cv2
import numpy as np

import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load mean vector 
mean_vec = np.load('Data/vectors/meanVector.npy') 

# Load eigen vectors 
eig_vec = np.load('Data/vectors/eigenVectors.npy')    

# Load weights
weights = np.load('Real-Time-Output/vectors/weights.npy')    

# Load faces 
facesDataset = np.load('Real-Time-Output/Vectors/facesVector.npy')                  

# Load identity
facesIdentity = np.load('Real-Time-Output/Vectors/identityVector.npy')				

cap = cv2.VideoCapture(0)

while(True):


    # Capture frame-by-frame
    ret, frame = cap.read()

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    

    for (x, y, w, h) in faces:
                
        # Crop face image
        face_img = gray[y+5:y+h-5, x+5:x+w-5]             
        # Resize to make uniform images
        face_img = cv2.resize(face_img, (47, 62))

        inputFace = face_img.reshape(1,-1)
        inputFace_weight = eig_vec @ (inputFace - mean_vec).T
        euclidean_distance = np.linalg.norm(weights - inputFace_weight, axis=0)
        best_match = np.argmin(euclidean_distance)

        if min(euclidean_distance) < 3000:
            print(euclidean_distance)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = facesIdentity[best_match]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            

        color = (255, 0, 0) #BGR 0-255 
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        
        
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()