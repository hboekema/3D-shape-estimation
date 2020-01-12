#!usr/bin/env/python3

import glob
import json
import os
from datetime import datetime

import cv2
import keras
import numpy as np
import segmentation_models as sm
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import argparse
import tensorflow as tf

#from data_generator import DataGenerator
from callbacks import PredOnEpochEnd


""" Set-up """

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to configuration file")
parser.add_argument("--silhouettes", help="Path to pre-generated silhouettes")
parser.add_argument("--run_id", help="Identifier of this network pass")

args = parser.parse_args()

# Read in the configurations
if args.config is not None:
    with open(args.config, 'r') as file:
        params = json.load(file)
else:
    with open("../config/server_network.json", 'r') as file:
        params = json.load(file)

# Set the ID of this training pass
if args.run_id is not None:
    run_id = args.run_id
else:
    # Use the current date and time as a run-id
    run_id = datetime.now().strftime("seggen_%Y-%m-%d_%H:%M:%S")

# Create experiment directory
exp_dir = "/data/cvfs/hjhb2/projects/segmentation_with_generator/experiments/" + str(run_id) + "/"
model_dir = exp_dir + "models/"
logs_dir = exp_dir + "logs/"
train_vis_dir = exp_dir + "train_vis/"
test_vis_dir = exp_dir + "test_vis/"
os.mkdir(exp_dir)
os.mkdir(model_dir)
os.mkdir(logs_dir)
os.mkdir(train_vis_dir)
os.mkdir(test_vis_dir)

# Set number of GPUs to use
#os.environ["CUDA_VISIBLE_DEVICES"] = params["ENV"]["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("gpu used: |" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")

# Set Keras format
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
#keras.backend.set_session(sess)
keras.backend.set_image_data_format(params["ENV"]["CHANNEL_FORMAT"])

# Set this to true for a more verbose output
tf.debugging.set_log_device_placement(False)

# Store the data paths
x_train_dir = params["DATA"]["SOURCE"]["TRAINING_IMG"]
y_train_dir = params["DATA"]["SOURCE"]["TRAINING_MASKS"]
x_test_dir = params["DATA"]["SOURCE"]["TEST_IMG"]

# Store the number of channels of the images and masks
img_n_channels = params["DATA"]["IMG_INFO"]["N_CHANNELS"]
mask_n_channels = params["DATA"]["MASK_INFO"]["N_CHANNELS"]

# Store the salient paths
#log_path = params["ENV"]["LOG_PATH"]
#train_pred_path = params["ENV"]["TRAINING_VIS_PATH"]
#test_pred_path = params["ENV"]["TEST_VIS_PATH"]

# Store the batch size
batch_size = params["GENERATOR"]["BATCH_SIZE"]

# Read in samples for evaluation
train_sample = np.array(cv2.cvtColor(cv2.imread(os.path.join(x_train_dir, "human/00001.png")), cv2.COLOR_BGR2RGB),
                        dtype='float32')
label_sample = np.array(cv2.imread(os.path.join(y_train_dir, "human/00001.png"), cv2.IMREAD_GRAYSCALE), dtype='float32')
test_sample = np.array(cv2.cvtColor(cv2.imread(os.path.join(x_test_dir, "VID_20170913_170437489_image_000001.png")),
                           cv2.COLOR_BGR2RGB), dtype='float32')

train_sample = train_sample.reshape((1, 256, 256, img_n_channels))/255
#label_sample = label_sample.reshape((1, 256, 256, 1))/255
test_sample = test_sample.reshape((1, 256, 256, img_n_channels))/255

"""

X_train = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(x_train_dir + "*.png")]
#X_train = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(x_train_dir + "*.png")]
print("Loaded training images")
Y_train = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(y_train_dir + "*.png")]
print("Loaded training labels")
X_test = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(x_test_dir + "*.png")]
print("Loaded test images")

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
X_train = X_train.reshape((X_train.shape[0], 256, 256, n_channels))

# Convert data type and normalize values
X_train = X_train.astype('float32')
X_train /= 255

Y_train = Y_train.reshape((Y_train.shape[0], 256, 256, 1))

# Convert data type and normalize values
Y_train = Y_train.astype('float32')
Y_train /= 255

X_test = X_test.reshape((X_test.shape[0], 256, 256, n_channels))

# Convert data type and normalize values
X_test = X_test.astype('float32')
X_test /= 255

"""

# Train the segmentation network
BACKBONE = params["MODEL"]["ARCHITECTURE"]["ENCODER"]
#preprocess_input = sm.get_preprocessing(BACKBONE)

# Sample inputs
#train_sample = preprocess_input(train_sample)
#test_sample = preprocess_input(test_sample)

# Test inputs
X_test = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(x_test_dir + "*.png")]
X_test = X_test[:20]
X_test = np.array(X_test, dtype='float32')
X_test = X_test.reshape((X_test.shape[0], 256, 256, img_n_channels))
X_test /= 255

"""
# preprocess input
x_train = preprocess_input(X_train)
y_train = preprocess_input(Y_train)
x_test = preprocess_input(X_test)

#plt.imshow(x_train[0])
#plt.show()
#plt.imshow(y_train[0])
#plt.show()
"""

# Parameters for generator class
#generator_params = {'dim': params["DATA"]["IMG_INFO"]["INPUT_WH"],
#          'batch_size': params["GENERATOR"]["BATCH_SIZE"],
#          'n_classes': params["DATA"]["MASK_INFO"]["N_CLASSES"],
#          'n_channels': img_n_channels,
#          'preprocessor': preprocess_input,
#          'shuffle': params["GENERATOR"]["SHUFFLE"]}

generator_params = {
    "rescale": 1./255,
    "shear_range": 0.2,
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "validation_split": 0.2,
    "dtype": "float32"
}

# Define generators
#training_generator = DataGenerator(x_train_dir, y_train_dir, **generator_params)
datagen_seed = 1
image_datagen = ImageDataGenerator(**generator_params)
mask_datagen = ImageDataGenerator(**generator_params)

x_train_generator = image_datagen.flow_from_directory(
        x_train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode="rgb",
        seed=datagen_seed,
        class_mode=None,
        subset="training"
)

y_train_generator = mask_datagen.flow_from_directory(
        y_train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode="grayscale",
        seed=datagen_seed,
        class_mode=None,
        subset="training"
)

x_val_generator = image_datagen.flow_from_directory(
        x_train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode="rgb",
        seed=datagen_seed,
        class_mode=None,
        subset="validation"
)

y_val_generator = mask_datagen.flow_from_directory(
        y_train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode="grayscale",
        seed=datagen_seed,
        class_mode=None,
        subset="validation"
)

training_generator = zip(x_train_generator, y_train_generator)
validation_generator = zip(x_val_generator, y_val_generator)

# Visualise some of the augmented images
#img_batch, mask_batch = next(training_generator)
#mask_batch = mask_batch.reshape(batch_size, 256, 256)
#for i in range(3):
#    fig, (ax1, ax2) = plt.subplots(1, 2)
#    fig.suptitle('Augmented image and mask')
#    ax1.imshow(img_batch[i])
#    ax2.imshow(mask_batch[i], cmap='gray')
#    plt.show()

#img_batch, mask_batch = next(validation_generator)
#mask_batch = mask_batch.reshape(batch_size, 256, 256)
#for i in range(3):
#    fig, (ax1, ax2) = plt.subplots(1, 2)
#    fig.suptitle('Augmented image and mask')
#    ax1.imshow(img_batch[i])
#    ax2.imshow(mask_batch[i], cmap='gray')
#    plt.show()

# Callback functions
# Create a model checkpoint after every epoch
model_save_checkpoint = ModelCheckpoint("../models/model.{epoch:02d}-{loss:.2f} " + str(run_id) + ".hdf5",
                                        monitor='loss', verbose=1, save_best_only=False, mode='auto',
                                        period=params["MODEL"]["CHKPT_PERIOD"])

# Stream the epoch loss to a file in JSON format. The file content
# is not well-formed JSON but rather has a JSON object per line.
#log_path = "../logs/loss_log_{}".format(datetime)
#epoch_log = open(log_path, mode='wt', buffering=1)
#json_logging_callback = LambdaCallback(
#    epoch_log=epoch_log,
#    x_test=x_test,
#    on_epoch_end=lambda epoch, logs, epoch_log, x_test: eval_epoch(epoch_log, x_test, epoch, logs),
#    on_train_end=lambda logs: epoch_log.close()
#)


#def eval_epoch(epoch_log, x_test, epoch, logs):
#    """ Write loss to a log and predict on sample images """
#    epoch_log.write(
#        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n')
#    cv2.imwrite("../training_visualisations/test_epoch_{}.png".format(epoch), model.predict(x_test[1]))


# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(None, None, img_n_channels))
model.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])


print("Fitting model...")
# Fit model
model.fit_generator(
    generator=training_generator,
    epochs=params["MODEL"]["EPOCHS"],
    steps_per_epoch=params["MODEL"]["STEPS_PER_EPOCH"],
    validation_data=validation_generator,
    validation_steps=params["MODEL"]["VALIDATION_STEPS"],
    callbacks=[PredOnEpochEnd(log_dir, x_train=train_sample, x_test=test_sample,
                              pred_path=train_vis_dir, run_id=run_id), model_save_checkpoint],
    use_multiprocessing=params["ENV"]["USE_MULTIPROCESSING"]
)

# Save the final model's weights
model.save("../models/final_model[{}].hdf5".format(run_id))

print("Model fit. Predicting...")
preds = model.predict(X_test)

preds = preds.reshape(preds.shape[0], 256, 256)
preds *= 255
preds.astype(np.uint8)

for i, pred in enumerate(preds, 1):
    cv2.imwrite(os.path.join(test_vis_dir, "img_{}_pred[{}].png".format(i, run_id)), pred)

plt.imshow(preds[0], cmap='gray')
plt.show()
