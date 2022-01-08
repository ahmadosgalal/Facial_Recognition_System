import numpy as np
import matplotlib.pyplot as plt
import cv2

# Haar cascade classifier to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load mean vector 
mean_vec = np.load('Data/vectors/meanVector.npy') 

# Load eigen vectors 
eig_vec = np.load('Data/vectors/eigenVectors.npy')    

# Load weights
weights = np.load('Data/vectors/weights.npy')    

# Load faces vector
facesDataset = np.load('Data/vectors/facesVector.npy')                   

# Load identity vector 
identity = np.load('Data/vectors/identityVector.npy') 

# Load X_train
X_train = np.load('Data/vectors/X_train.npy')                   

# Load y_train
y_train = np.load('Data/vectors/y_train.npy') 



# Load X_train
X_test = np.load('Data/vectors/X_test.npy')                   

# Load y_test
y_test = np.load('Data/vectors/y_test.npy') 

accuracy = 0
n_samples, height, width = X_test.shape
faceshape = (height, width)


for i in range(n_samples):
    # Test on out-of-sample image of existing class
    image = X_test[i]
 
    # Detect faces in image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.5, minNeighbors=5)

    # If there was faces detected in image
    if(len(faces) != 0):
        # For each detected face
        for (x, y, w, h) in faces:
            # Crop face image
            face_img = image[y+5:y+h-5, x+5:x+w-5]             
            # Resize to make uniform images
            face_img = cv2.resize(face_img, (47, 62))          
            # Save the image   
            image = face_img


    # Visualize
    #fig, axes = plt.subplots(1,1,sharex=True,sharey=True,figsize=(8,6))
    #axes.imshow(image, cmap="gray")

    testImage = image.reshape(1,-1)

    testImage_weight = eig_vec @ (testImage - mean_vec).T
    euclidean_distance = np.linalg.norm(weights - testImage_weight, axis=0)
    best_match = np.argmin(euclidean_distance)
    top_match = np.argsort(euclidean_distance)[:7]

    if(y_test[i] == y_train[best_match]):
        accuracy += 1
    #print(testImage_weight.shape, euclidean_distance.shape)
    #print(best_match)

    #print(top_match)
    print("Person: " , y_test[i])
    print("Best match %s with Euclidean distance %f" % (y_train[best_match], euclidean_distance[best_match]))

    print("Second best match %s with Euclidean distance %f" % (y_train[top_match[1]], euclidean_distance[top_match[1]]))
    print("Third best match %s with Euclidean distance %f" % (y_train[top_match[2]], euclidean_distance[top_match[2]]))
    print("Fourth best match %s with Euclidean distance %f" % (y_train[top_match[3]], euclidean_distance[top_match[3]]))
    print("Fifth best match %s with Euclidean distance %f" % (y_train[top_match[4]], euclidean_distance[top_match[4]]))
    print("Sixth best match %s with Euclidean distance %f" % (y_train[top_match[5]], euclidean_distance[top_match[5]]))
    print("Seventh best match %s with Euclidean distance %f" % (y_train[top_match[6]], euclidean_distance[top_match[6]]))

    print("\n")

print(accuracy/n_samples)

'''
    # Visualize
    fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
    axes[0].imshow(testImage.reshape(height, width), cmap="gray")
    axes[0].set_title("Test Image: " + str(y_test[i]))
    axes[1].imshow(X_train[best_match].reshape(faceshape), cmap="gray")
    axes[1].set_title("Best Match: " + str(y_train[best_match]))
    plt.show()
'''