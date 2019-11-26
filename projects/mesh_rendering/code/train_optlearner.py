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
import pickle

#from locoptlearner import optlearnerArchitecture, mesh_mse
from optlearner import OptLearnerArchitecture, OptLearnerDistArchitecture, false_loss, no_loss
from smpl_np import SMPLModel
from render_mesh import Mesh
from callbacks import PredOnEpochEnd, OptLearnerPredOnEpochEnd
from silhouette_generator import SilhouetteDataGenerator, OptLearnerDataGenerator


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
os.environ["CUDA_VISIBLE_DEVICES"] = params["ENV"]["CUDA_VISIBLE_DEVICES"]
print("gpu used:|" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")
#exit(1)

# Ensure that TF2.0 is not used
#tf.disable_v2_behavior()
#tf.enable_eager_execution()

# Set Keras format
tf.keras.backend.set_image_data_format(params["ENV"]["CHANNEL_FORMAT"])

# Store the data paths
#train_dir = params["DATA"]["SOURCE"]["TRAIN"]
#val_dir = params["DATA"]["SOURCE"]["VAL"]
#test_dir = params["DATA"]["SOURCE"]["TEST"]
train_dir = "../data/train/"
val_dir = "../data/val/"
test_dir = "../data/test/"

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
train_gen = OptLearnerDataGenerator(train_dir, batch_size=batch_size, debug=False)
val_gen = OptLearnerDataGenerator(train_dir, batch_size=batch_size)   # use same data as for training, but vary the position of the arm by a different amount
test_gen = OptLearnerDataGenerator(test_dir, batch_size=batch_size)

# Get a sample train input for the prediction callback
train_ids = os.listdir(train_dir)[:2]
train_sample_indices = []
train_sample_params = []
train_sample_pcs = []
train_sample_silh = []
for i, sample in enumerate(train_ids):
    with open(os.path.join(train_dir, sample), 'r') as handle:
        data_dict = pickle.load(handle)
    train_sample_indices.append(i)
    train_sample_params.append(data_dict["parameters"])
    train_sample_pcs.append(data_dict["pointcloud"])
    train_sample_silh.append(data_dict["silhouette"])

train_sample_input = [np.array(train_sample_indices), np.array(train_sample_params), np.array(train_sample_pcs)]

test_ids = os.listdir(test_dir)[:2]
test_sample_indices = []
test_sample_params = []
test_sample_pcs = []
test_sample_silh = []
for i, sample in enumerate(test_ids):
    with open(os.path.join(test_dir, sample), 'r') as handle:
        data_dict = pickle.load(handle)
    test_sample_indices.append(i)
    test_sample_params.append(data_dict["parameters"])
    test_sample_pcs.append(data_dict["pointcloud"])
    test_sample_silh.append(data_dict["silhouette"])

test_sample_input = [np.array(test_sample_indices), np.array(test_sample_params), np.array(test_sample_pcs)]


""" Model set-up """

# Callback functions
# Create a model checkpoint after every few epochs
model_save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_dir + "model.{epoch:02d}-{loss:.2f}.hdf5",
    monitor='loss', verbose=1, save_best_only=False, mode='auto',
    period=save_period, save_weights_only=True)

# Predict on sample params at the end of every few epochs
#epoch_pred_cb = PredOnEpochEnd(logs_dir, smpl, x_train=train_sample_x, y_train=train_sample_pc, x_test=sample_x, y_test=sample_pc,
#                               pred_path=train_vis_dir, period=pred_period, visualise=False)
epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=train_sample_input, train_silh=train_sample_silh, test_inputs=test_sample_input, test_silh=test_sample_silh,
        pred_path=train_vis_dir, period=pred_period, visualise=False)

# Make model entity
# Create initializer for the embedding layer
np.random.seed(10)
offset = np.zeros(85)
offset[42:45] = np.random.rand(3) # change left shoulder 1
offset[51:54] = np.random.rand(3) # change left shoulder 2
offset[57:60] = np.random.rand(3) # change left elbow
offset[63:66] = np.random.rand(3) # change left wrist
offset[69:72] = np.random.rand(3) # change left fingers
#offset[72:75] = np.random.rand(3) # change global translation
embedding_initializer = train_gen.yield_params(offset="right arm")

optlearner_inputs, optlearner_outputs = OptLearnerArchitecture(embedding_initializer)
#optlearner_inputs, optlearner_outputs = OptLearnerDistArchitecture(embedding_initializer)
print("optlearner inputs " +str(optlearner_inputs))
print("optlearner outputs "+str(optlearner_outputs))
optlearner_model = Model(inputs=optlearner_inputs, outputs=optlearner_outputs)
optlearner_model.summary()

learning_rate = 0.001
optlearner_model.compile(optimizer=Adam(lr=learning_rate, decay=0.0001), loss=[no_loss, false_loss, no_loss, false_loss, false_loss, false_loss],
        loss_weights=[0.0, 1.0, 0.0, 1.0, 1.0, 0.1])

""" Training loop"""

X_batch, Y_batch = train_gen.__getitem__(0)
X_index = X_batch[0]
X_params = X_batch[1]
X_pc = X_batch[2]
#print("X_batch first index value: " + str(X_index))
print("X_batch index shape: " + str(X_index.shape))
print("X_batch parameters shape: " + str(X_params.shape))
print("X_batch point cloud shape: " + str(X_pc.shape))
Y_batch_1 = Y_batch[0]
Y_batch_2 = Y_batch[1]
Y_batch_3 = Y_batch[2]
Y_batch_4 = Y_batch[3]
Y_batch_5 = Y_batch[4]
Y_batch_6 = Y_batch[5]
print("Y optlearner_params shape: " + str(Y_batch_1.shape))
print("Y delta_d_loss shape: " + str(Y_batch_2.shape))
print("Y optlearner_pc shape: " + str(Y_batch_3.shape))
print("Y pointcloud_loss shape: " + str(Y_batch_4.shape))
print("Y delta_d_hat_loss shape: " + str(Y_batch_5.shape))
print("Y smpl_loss shape: " + str(Y_batch_6.shape))

# Run the main loop
optlearner_model.fit_generator(
    generator=train_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=[epoch_pred_cb, model_save_checkpoint],
    #use_multiprocessing=params["ENV"]["USE_MULTIPROCESSING"]
)

# Store the model
print("Saving model to " + str(model_dir) + "model.final.hdf5...")
optlearner_model.save_weights(model_dir + "model.final.hdf5")

#print("Loading model...")
#loaded_inputs, loaded_outputs = OptLearnerArchitecture()
#loaded_model = Model(inputs=loaded_inputs, outputs=loaded_outputs)
#loaded_model.load_weights(model_dir + "model.final.hdf5")
#loaded_model.summary()
#loaded_model.compile(optimizer=Adam(lr=learning_rate, decay=0.0001), loss=[no_loss, false_loss, no_loss, false_loss, false_loss, false_loss],
#                 loss_weights=[0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

