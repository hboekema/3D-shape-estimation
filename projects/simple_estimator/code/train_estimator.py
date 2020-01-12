import os
import argparse
import json
from datetime import datetime
import keras
from keras.callbacks import LambdaCallback
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.losses import binary_crossentropy
#import segmentation_models as sm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import cv2
#import tensorflow.compat.v1 as tf
import tensorflow as tf
from smpl_np import SMPLModel
import pickle
from itertools import izip

from architectures import SimpleArchitecture, no_loss, false_loss
from keras.preprocessing.image import ImageDataGenerator
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
    run_id = datetime.now().strftime("mesh_normal_optimiser_%Y-%m-%d_%H:%M:%S")

# Create experiment directory
exp_dir = "/data/cvfs/hjhb2/projects/simple_estimator/experiments/" + str(run_id) + "/"
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
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"  #params["ENV"]["CUDA_VISIBLE_DEVICES"]
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"  #params["ENV"]["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #params["ENV"]["CUDA_VISIBLE_DEVICES"]
print("gpu used:|" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")
#exit(1)

# Ensure that TF2.0 is not used
#tf.disable_v2_behavior()
#tf.enable_eager_execution()

# Set Keras format
tf.keras.backend.set_image_data_format(params["ENV"]["CHANNEL_FORMAT"])


# Store the batch size and number of epochs
batch_size = 32
#batch_size = params["GENERATOR"]["BATCH_SIZE"]
epochs = params["MODEL"]["EPOCHS"]
steps_per_epoch = params["MODEL"]["STEPS_PER_EPOCH"]
validation_steps = params["MODEL"]["VALIDATION_STEPS"]
#save_period = params["MODEL"]["SAVE_PERIOD"]
#pred_period = params["MODEL"]["PRED_PERIOD"]

# Store the data paths
x_train_dir = params["DATA"]["SOURCE"]["TRAINING_IMG"]
y_train_dir = params["DATA"]["SOURCE"]["TRAINING_MASKS"]
x_test_dir = params["DATA"]["SOURCE"]["TEST_IMG"]

# Store the number of channels of the images and masks
#img_n_channels = params["DATA"]["IMG_INFO"]["N_CHANNELS"]
#mask_n_channels = params["DATA"]["MASK_INFO"]["N_CHANNELS"]
mask_n_channels = 1

""" Data collection """

np.random.seed(10)

# Read in samples for evaluation
label_sample = np.array(cv2.imread(os.path.join(y_train_dir, "human/00001.png"), cv2.IMREAD_GRAYSCALE), dtype='float32')
label_sample = label_sample.reshape((1, 256, 256, 1))/255


print("Preparing generators...")
generator_params = {
    "rescale": 1./255,
    "shear_range": 0.2,
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "validation_split": 0.2,
#    "dtype": "float32"
}

# Define generators
#training_generator = DataGenerator(x_train_dir, y_train_dir, **generator_params)
datagen_seed = 1
mask_datagen = ImageDataGenerator(**generator_params)

y_train_generator = mask_datagen.flow_from_directory(
        y_train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        color_mode="grayscale",
        seed=datagen_seed,
        class_mode=None,
        subset="training"
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

print("Zipping generators...")
training_generator = izip(y_train_generator, y_train_generator)
validation_generator = izip(y_val_generator, y_val_generator)

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



""" Model set-up  """

# SMPL model to be used
smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/./basicModel_f_lbs_10_207_0_v1.0.0.pkl')

# Callback functions
# Create a model checkpoint after every few epochs
SAVE_PERIOD = 100
model_save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_dir + "model.{epoch:02d}-{loss:.2f}.hdf5",
    monitor='loss', verbose=1, save_best_only=False, mode='auto',
    period=SAVE_PERIOD, save_weights_only=True)

# Predict on sample params at the end of every few epochs
PERIOD = 25
pred_cb = PredOnEpochEnd(logs_dir, smpl, x_train=label_sample, pred_path=train_vis_dir)

# Build and compile the model
estimator_inputs, estimator_outputs = SimpleArchitecture(input_shape=(256, 256, 1))
print("estimator inputs " +str(estimator_inputs))
print("estimator outputs "+str(estimator_outputs))
estimator_model = Model(inputs=estimator_inputs, outputs=estimator_outputs)
estimator_model.summary()

learning_rate = 0.001
optimizer= Adam(lr=learning_rate, decay=0.0001, clipvalue=100.0)
#optimizer= SGD(lr=learning_rate)
estimator_model.compile(optimizer=optimizer, loss=[no_loss, no_loss, binary_crossentropy],
                                            loss_weights=[0.0, 0.0, 1.0]
                                            )


""" Training loop"""


estimator_model.fit_generator(
    generator=training_generator,
    epochs=10000,
    steps_per_epoch=16,
    validation_data=validation_generator,
    validation_steps=8,
    callbacks=[pred_cb, model_save_checkpoint],
    use_multiprocessing=True
)

# Store the model
print("Saving model to " + str(model_dir) + "model.final.hdf5...")
estimator_model.save_weights(model_dir + "model.final.hdf5")

