import numpy as np
import matplotlib.pyplot as plt
import cv2

# Haar cascade classifier to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load mean vector 
mean_vec = np.load('real-time/vectors/meanVector.npy') 

# Load eigen vectors 
eig_vec = np.load('real-time/vectors/eigenVectors.npy')    

# Load weights
weights = np.load('real-time/vectors/weights.npy')    

# Load X_train
X_train = np.load('real-time/vectors/X_train.npy')                   

# Load y_train
y_train = np.load('real-time/vectors/y_train.npy') 

print(y_train)


n_samples, height, width = X_train.shape
faceshape = (height, width)
print(faceshape)


# Test on out-of-sample image of existing class
image = cv2.imread("abdullah_Gul.jpg")

# Convert image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)         
            
# Detect faces in image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=5)

# If there was faces detected in image
if(len(faces) != 0):
    # For each detected face
    for (x, y, w, h) in faces:
        # Crop face image
        face_img = gray_image[y+5:y+h-5, x+5:x+w-5]             
        # Resize to make uniform images
        face_img = cv2.resize(face_img, (47, 62))          
        # Save the image   
        image = face_img


# Visualize
fig, axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(8,6))
axes.imshow(image, cmap="gray")

testImage = image.reshape(1,-1)

testImage_weight = eig_vec @ (testImage - mean_vec).T
euclidean_distance = np.linalg.norm(weights - testImage_weight, axis=0)
best_match = np.argmin(euclidean_distance)
top_match = np.argsort(euclidean_distance)[:7]
print(testImage_weight.shape, euclidean_distance.shape)
#print(best_match)

#print(top_match)

print("Best match %s with Euclidean distance %f" % (y_train[best_match], euclidean_distance[best_match]))

print("Second best match %s with Euclidean distance %f" % (y_train[top_match[1]], euclidean_distance[top_match[1]]))
print("Third best match %s with Euclidean distance %f" % (y_train[top_match[2]], euclidean_distance[top_match[2]]))
print("Fourth best match %s with Euclidean distance %f" % (y_train[top_match[3]], euclidean_distance[top_match[3]]))
print("Fifth best match %s with Euclidean distance %f" % (y_train[top_match[4]], euclidean_distance[top_match[4]]))
print("Sixth best match %s with Euclidean distance %f" % (y_train[top_match[5]], euclidean_distance[top_match[5]]))
print("Seventh best match %s with Euclidean distance %f" % (y_train[top_match[6]], euclidean_distance[top_match[6]]))

# Visualize
fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
axes[0].imshow(testImage.reshape(height, width), cmap="gray")
axes[0].set_title("Test Image")
axes[1].imshow(X_train[best_match].reshape(faceshape), cmap="gray")
axes[1].set_title("Best Match")
plt.show()
