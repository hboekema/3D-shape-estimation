#!usr/bin/env/python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import segmentation_models as sm
import keras
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

keras.backend.set_image_data_format('channels_last')

data_dir = './data/'

# Load the test and training data
x_train_dir = os.path.join(data_dir, 'train_img/')
y_train_dir = os.path.join(data_dir, 'train_labels/')
x_test_dir = os.path.join(data_dir, 'test_img/')

X_train = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(x_train_dir + "*.png")]
#X_train = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(x_train_dir + "*.png")]
print("Loaded training images")
Y_train = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(y_train_dir + "*.png")]
print("Loaded training labels")
X_test = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(x_test_dir + "*.png")]
print("Loaded test images")

X_train = X_train[:200]
Y_train = Y_train[:200]
X_test = X_test[:20]

len_X = len(X_train)
len_Y = len(Y_train)

if len_X > len_Y:
    X_train = X_train[:len_Y]
elif len_Y > len_X:
    Y_train = Y_train[:len_X]

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)

n_channels = 3
X_train = X_train.reshape(X_train.shape[0], 256, 256, n_channels)

# Convert data type and normalize values
X_train = X_train.astype('float32')
X_train /= 255

Y_train = Y_train.reshape(Y_train.shape[0], 256, 256, 1)

# Convert data type and normalize values
Y_train = Y_train.astype('float32')
Y_train /= 255

X_test = X_test.reshape(X_test.shape[0], 256, 256, n_channels)

# Convert data type and normalize values
X_test = X_test.astype('float32')
X_test /= 255

# Train the segmentation network
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
x_train = preprocess_input(X_train)
y_train = preprocess_input(Y_train)
x_test = preprocess_input(X_test)

#plt.imshow(x_train[0])
#plt.show()
#plt.imshow(y_train[0])
#plt.show()

# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(None, None, n_channels))
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

print("Fitting model...")
# Fit model
model.fit(
    x=np.array(x_train),
    y=np.array(y_train),
    batch_size=32,
    epochs=1
)

print("Model fit. Predicting...")
preds = model.predict(x_test)

preds = preds.reshape(preds.shape[0], 256, 256)
print(preds.shape)

for i, pred in enumerate(preds, 1):
    #plt.imshow(pred, cmap='gray')
    #plt.show()
    cv2.imwrite("img_pred_{}.png".format(i), pred)
