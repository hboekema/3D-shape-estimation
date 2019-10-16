#!usr/bin/env/python3

from __future__ import print_function

import numpy as np
np.random.seed(123)   # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print shape of the MNIST dataset - should be (60000, 28, 28)
print(X_train.shape)

# Graph the first sample of the dataset for confirmation of the above
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.imshow(X_train[0])
plt.show()

# Reshape dataset to specify that there is a single channel
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Convert data type and normalize values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Print shape of training labels - should be (60000,)
print(y_train.shape)

# Fix this by turning the labels into 10-dimensional class matrices
# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Now it should be (60000, 10)
print(Y_train.shape)

# Build the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Define loss function and optimiser
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model to the training data
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)

# Evaluate the model on the test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
model
