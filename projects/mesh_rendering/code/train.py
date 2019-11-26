import os
import argparse
import json
from datetime import datetime
import keras
from keras.models import Model
from keras.optimizers import Adam

import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf

from locencoder import EncoderArchitecture, mesh_mse
from smpl_np import SMPLModel
from render_mesh import Mesh
from callbacks import PredOnEpochEnd
from silhouette_generator import SilhouetteDataGenerator


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
    run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Create experiment directory
exp_dir = "../experiments/" + str(run_id) + "/"
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
os.environ["CUDA_VISIBLE_DEVICES"] = "6" # params["ENV"]["CUDA_VISIBLE_DEVICES"]
print("gpu used:|" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")
#exit(1)

# Ensure that TF2.0 is not used
#tf.disable_v2_behavior()
#tf.enable_eager_execution()

# Set Keras format
tf.keras.backend.set_image_data_format(params["ENV"]["CHANNEL_FORMAT"])

# Store the data paths
train_dir = params["DATA"]["SOURCE"]["TRAIN"]
val_dir = params["DATA"]["SOURCE"]["VAL"]
test_dir = params["DATA"]["SOURCE"]["TEST"]

# Store the width, height and number of channels of the silhouettes
silh_dim = params["DATA"]["SILH_INFO"]["INPUT_WH"]
silh_n_channels = params["DATA"]["SILH_INFO"]["N_CHANNELS"]

# Store the batch size and number of epochs
batch_size = params["GENERATOR"]["BATCH_SIZE"]
epochs = params["MODEL"]["EPOCHS"]
steps_per_epoch = params["MODEL"]["STEPS_PER_EPOCH"]
validation_steps = params["MODEL"]["VALIDATION_STEPS"]
save_period = params["MODEL"]["SAVE_PERIOD"]
pred_period = params["MODEL"]["PRED_PERIOD"]

""" Data generation """

# Generate the silhouettes from the SMPL parameters
smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/./basicModel_f_lbs_10_207_0_v1.0.0.pkl')

# Load the SMPL data
#train_gen = SilhouetteDataGenerator(train_dir, smpl, batch_size=batch_size, img_dim=silh_dim, frac_randomised=1.0, debug=True)
#val_gen = SilhouetteDataGenerator(val_dir, smpl, batch_size=batch_size, img_dim=silh_dim, frac_randomised=1.0, debug=True)
#test_gen = SilhouetteDataGenerator(test_dir, smpl, batch_size=batch_size, img_dim=silh_dim, frac_randomised=1.0, noise=0.0)
train_gen = LoadedDataGenerator(train_dir, batch_size=batch_size)
val_gen = LoadedDataGenerator(val_dir, batch_size=batch_size)
test_gen = LoadedDataGenerator(test_dir, batch_size=batch_size)

# Get a sample train input for the prediction callback
X_batch, Y_batch = train_gen.__getitem__(0)
#print("X_batch shape: " +str(X_batch.shape))
#print("Y_batch shape: " +str(Y_batch.shape))
train_sample_x = X_batch[0].reshape(256, 256, 1).astype("float32")
train_sample_y = np.array(Y_batch[0][0])
train_sample_pc = np.array(Y_batch[1][0])
#print("Y sample params shape: " + str(train_sample_y.shape))
#print("Y sample pc shape: " + str(train_sample_pc.shape))

# Artificial sample data
sample_pose = 0.65 * (np.random.rand(smpl.pose_shape[0], smpl.pose_shape[1]) - 0.5)
sample_beta = 0.2 * (np.random.rand(smpl.beta_shape[0]) - 0.5)
sample_trans = np.zeros(smpl.trans_shape[0])
#sample_trans = 0.1 * (np.random.rand(smpl.trans_shape[0]) - 0.5)

sample_y = np.array([sample_pose.ravel(), sample_beta, sample_trans])
sample_pc = smpl.set_params(sample_pose, sample_beta, sample_trans)
sample_x = Mesh(pointcloud=sample_pc).render_silhouette(dim=silh_dim, show=False)
sample_x = sample_x.reshape((silh_dim[0], silh_dim[1], silh_n_channels)).astype("float32")
sample_x /= 255

""" Model set-up """

# Callback functions
# Create a model checkpoint after every few epochs
#model_save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
#    model_dir + "model.{epoch:02d}-{loss:.2f}.hdf5",
#    monitor='loss', verbose=1, save_best_only=False, mode='auto',
#    period=save_period, save_weights_only=True)

model_save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_dir + "model.{epoch:02d}-{loss:.2f}.hdf5",
    monitor='loss', verbose=1, save_best_only=False, mode='auto',
    period=100, save_weights_only=True)

# Predict on sample images at the end of every few epochs
epoch_pred_cb = PredOnEpochEnd(logs_dir, smpl, x_train=train_sample_x, y_train=train_sample_pc, x_test=sample_x, y_test=sample_pc,
                               pred_path=train_vis_dir, period=pred_period, visualise=False)


# Make model entity
encoder_inputs, encoder_outputs = EncoderArchitecture((silh_dim[0], silh_dim[1], silh_n_channels))
print("encoder inputs " +str(encoder_inputs))
print("encoder outputs "+str(encoder_outputs))
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_outputs)
encoder_model.summary()

encoder_model.compile(optimizer=Adam(lr=0.001, decay=0.001), loss=['mse', mesh_mse], loss_weights=[1.0, 1.0])
#encoder_model.compile(optimizer=Adam(lr=0.001, decay=0.001), loss=['mse', mesh_mse], loss_weights=[1.0, 1.0])
#encoder_model.compile(optimizer='adam', loss='mse')
#epoch_pred_cb.set_model(encoder_model)

""" Training loop"""

X_batch, Y_batch = train_gen.__getitem__(0)
#X_batch_1 = X_batch[0].reshape(1 ,256, 256, 1)
#Y_batch_1 = Y_batch[0].reshape(1, 85)
print("X_batch first value: " + str(X_batch.shape))
print("Y_batch first param value: " + str(Y_batch[0].shape))
print("Y_batch first pc value: " + str(Y_batch[1].shape))
#encoder_model.fit(
#        X_batch,
#        Y_batch,
#        epochs=100
#)

# Run the main loop
encoder_model.fit_generator(
    generator=train_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=[epoch_pred_cb, model_save_checkpoint],
    #use_multiprocessing=params["ENV"]["USE_MULTIPROCESSING"]
)

# Store the model
encoder_model.save_weights(model_dir + "model.final.hdf5")

