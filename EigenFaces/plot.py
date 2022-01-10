import numpy as np
import matplotlib.pyplot as plt

# Visualise portraits
def plot_portraits(images, titles, h, w, n_row, n_col):
    fig, axes = plt.subplots(n_row, n_col ,sharex=True, sharey=True, figsize=(8, 6))

    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i].reshape(h, w), cmap="gray")
        ax.set_title(titles[i])

    plt.show()


# Load eigen faces 
eigenFaces = np.load('Data/vectors/eigenFaces.npy') 

 # Load faces vector
faces = np.load('Data/vectors/facesVector.npy')                   

# Load identity vector 
identity = np.load('Data/vectors/identityVector.npy') 

# Load mean vector 
mean_vec = np.load('Data/vectors/meanVector.npy') 


n_samples, h, w = faces.shape


# Show first 10 eigen faces

eigenface_titles = ["eigenface %d" % i for i in range(eigenFaces.shape[0])]
plot_portraits(eigenFaces, eigenface_titles, h, w, 10, 10) 



# Show first 10 faces
face_titles = ["Face %d" % i for i in range(faces.shape[0])]
plot_portraits(faces, face_titles, faces.shape[1], faces.shape[2], 10, 10) 


# Show mean face
mean_face = mean_vec.reshape(h, w)
fig, axes = plt.subplots(1, 1 ,sharex=True, sharey=True, figsize=(8, 6))
axes.imshow(mean_face, cmap="gray")
plt.show()

