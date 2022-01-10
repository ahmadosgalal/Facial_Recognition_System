# Import essential libraries
import requests
import cv2
import numpy as np
import imutils
  



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load mean vector 
mean_vec = np.load('Real-Time-Output/vectors/meanVector.npy') 

# Load eigen vectors 
eig_vec = np.load('Data/vectors/eigenVectors.npy')    

# Load weights
weights = np.load('Real-Time-Output/vectors/weights.npy')    

# Load faces 
facesDataset = np.load('Real-Time-Output/Vectors/facesVector.npy')                  

# Load identity
facesIdentity = np.load('Real-Time-Output/Vectors/identityVector.npy')			

# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://192.168.137.182:8080/shot.jpg"
  
# While loop to continuously fetching data from the Url
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=460, height=960)
    
    #vs = cv2.VideoCapture(url + "/video")


    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    

    for (x, y, w, h) in faces:
                
        # Crop face image
        face_img = gray[y+5:y+h-5, x+5:x+w-5]             
        # Resize to make uniform images
        face_img = cv2.resize(face_img, (32, 32))

        inputFace = face_img.reshape(1,-1)
        inputFace_weight = eig_vec @ (inputFace - mean_vec).T
        euclidean_distance = np.linalg.norm(weights - inputFace_weight, axis=0)
        best_match = np.argmin(euclidean_distance)
        #print(facesIdentity[best_match])

        if min(euclidean_distance) < 2000:
            #print(min(euclidean_distance))
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = facesIdentity[best_match]
            color = (255, 0, 0)
            stroke = 2
            cv2.putText(img, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            #print("Face recognized: ", name)

        #else:
            #print("Face not recognized")

        color = (255, 0, 0) #BGR 0-255 
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)

        cv2.imshow("Android_cam", img)

    # Press Esc key to exit
    if cv2.waitKey(1) == 27:
        break
  
cv2.destroyAllWindows()