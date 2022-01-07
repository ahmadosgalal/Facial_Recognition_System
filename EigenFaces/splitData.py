import numpy as np
from sklearn.model_selection import train_test_split

# Load faces vector
X = np.load('real-time/vectors/X.npy')                   

# Load identity vector 
y = np.load('real-time/vectors/y.npy') 


# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


np.save('real-time/vectors/X_train', X_train)               # Store X_train
np.save('real-time/vectors/y_train', y_train)				# Store y_train


np.save('real-time/vectors/X_test', X_test)                 # Store X_test
np.save('real-time/vectors/y_test', y_test)				# Store y_test
