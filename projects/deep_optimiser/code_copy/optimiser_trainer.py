import os
import argparse
import json
from datetime import datetime
import keras
from keras.callbacks import LambdaCallback
from keras.models import Model
from keras.optimizers import Adam,SGD
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import pickle

from optlearner import OptLearnerStaticModArchitecture, false_loss, no_loss, load_smpl_params, emb_init_weights
from smpl_np import SMPLModel
from render_mesh import Mesh
from callbacks import PredOnEpochEnd, OptLearnerPredOnEpochEnd
#from silhouette_generator import SilhouetteDataGenerator, OptLearnerExtraOutputDataGenerator, OLExtraOutputZeroPoseDataGenerator


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
    run_id = datetime.now().strftime("diff_vec_optimiser_%Y-%m-%d_%H:%M:%S")

# Create experiment directory
exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/" + str(run_id) + "/"
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
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
#train_dir = "../data/train/"
#val_dir = "../data/val/"
#test_dir = "../data/test/"

# Store the batch size and number of epochs
batch_size = params["GENERATOR"]["BATCH_SIZE"]
epochs = params["MODEL"]["EPOCHS"]
steps_per_epoch = params["MODEL"]["STEPS_PER_EPOCH"]
validation_steps = params["MODEL"]["VALIDATION_STEPS"]
save_period = params["MODEL"]["SAVE_PERIOD"]
pred_period = params["MODEL"]["PRED_PERIOD"]

""" Data generation """

np.random.seed(10)

param_ids = ["param_{:02d}".format(i) for i in range(85)]
#not_trainable = [0, 1, 2]
#trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 70]
#trainable_params = [param_ids[index] for index in trainable_params_indices]

#trainable_params = ["param_00", "param_01", "param_02", "param_56", "param_57", "param_58", "param_59", "param_60", "param_61"]
trainable_params = ["param_01", "param_59", "param_56"]
#trainable_params = ["param_59", "param_56"]
param_trainable = { param: (param in trainable_params) for param in param_ids }
# Generate the data from the SMPL parameters
smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
zero_params = np.zeros(shape=(85,))
zero_pc = smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85])
#print("zero_pc: " + str(zero_pc))
#base_params = 0.2 * (np.random.rand(85) - 0.5)
#base_pc = smpl.set_params(beta=base_params[72:82], pose=base_params[0:72].reshape((24,3)), trans=base_params[82:85])


data_samples = 10000
#data_samples = 1000
#data_samples = 100
#data_samples = 10
X_indices = np.array([i for i in range(data_samples)])
X_params = 0.2 * 2*(np.random.rand(data_samples, 85) - 0.5)
#X_params = np.array([zero_params for i in range(data_samples)], dtype="float64")


X_params[:, 56] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 57] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 58] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
X_params[:, 59] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 60] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 61] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 0] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
X_params[:, 1] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_params[:, 2] = 2 * np.pi * (np.random.rand(data_samples) - 0.5)
#X_pcs = np.array([zero_pc for i in range(data_samples)], dtype="float64")
#X_params = np.array([base_params for i in range(data_samples)], dtype="float32")
#X_pcs = np.array([base_pc for i in range(data_samples)], dtype="float32")

#X_params = 0.2 * np.random.rand(data_samples, 85)
X_pcs = []
for params in X_params:
    pc = smpl.set_params(beta=params[72:82], pose=params[0:72], trans=params[82:85])
    X_pcs.append(pc)
X_pcs = np.array(X_pcs)
X_data = [np.array(X_indices), np.array(X_params), np.array(X_pcs)]
Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85))]
#Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 7))]
#Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 35)), np.zeros((data_samples, 35)), np.zeros((data_samples, 35))]

# Render silhouettes for the callback data
num_samples = 3
cb_indices = X_indices[:num_samples]
cb_params = X_params[:num_samples]
cb_pcs = X_pcs[:num_samples]
X_cb = [np.array(cb_indices), np.array(cb_params), np.array(cb_pcs)]
silh_cb = []
for pc in cb_pcs:
    silh = Mesh(pointcloud=pc).render_silhouette(show=False)
    silh_cb.append(silh)

""" Model set-up """

# Embedding initialiser
emb_initialiser = emb_init_weights(X_params)

# Callback functions
# Create a model checkpoint after every few epochs
SAVE_PERIOD = 50
model_save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_dir + "model.{epoch:02d}-{delta_d_mse_loss:.2f}.hdf5",
    monitor='loss', verbose=1, save_best_only=False, mode='auto',
    period=SAVE_PERIOD, save_weights_only=True)

# Predict on sample params at the end of every few epochs
PERIOD = 1
epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PERIOD, trainable_params=trainable_params, visualise=False)


# Build and compile the model
smpl_params, input_info, faces = load_smpl_params()
optlearner_inputs, optlearner_outputs = OptLearnerStaticModArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, emb_size=data_samples, smpl_params=smpl_params, input_info=input_info, faces=faces)
print("optlearner inputs " +str(optlearner_inputs))
print("optlearner outputs "+str(optlearner_outputs))
optlearner_model = Model(inputs=optlearner_inputs, outputs=optlearner_outputs)
optlearner_model.summary()

# Model options
#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata= tf.RunMetadata()

learning_rate = 0.001 #0.001
optimizer= Adam(lr=learning_rate, decay=0.0001, clipvalue=10000.0)
#optimizer= Adam(lr=learning_rate, decay=0.0)
#optimizer= SGD(lr=learning_rate, decay=0.0001, clipvalue=100.0)
optlearner_model.compile(optimizer=optimizer, loss=[no_loss, false_loss, no_loss,
                                                   false_loss,
                                                   false_loss,
                                                   false_loss,
						   false_loss, #no_loss, #false_loss,  #this is the loss which updates smpl parameter inputs with predicted gradient
						   no_loss, no_loss, no_loss,
                                                   #no_loss, no_loss, no_loss
                                                   ],
                                            loss_weights=[0.0,
                                                        1.0*10, # delta_d loss
                                                        0.0,
                                                        1.0*10, # point cloud loss
                                                        1.0*10, # delta_d_hat loss
                                                        0.0, # delta_d_hat sin loss
                                                        0.0/learning_rate, # this is the loss which updates smpl parameter inputs with predicted gradient
                                                        0.0, 0.0, 0.0,
                                                        #0.0, 0.0, 0.0
                                                        ],
                                            #options=run_options,
                                            #run_metadata=run_metadata
                                            )

# Initialise the embedding layer weights
#initial_weights = emb_init(shape=(1, 1000, 85))
#emb_layer = optlearner_model.get_layer("parameter_embedding")
#emb_layer.set_weights(initial_weights)

""" Training loop"""

print("X index shape: " + str(X_indices.shape))
print("X parameters shape: " + str(X_params.shape))
print("X point cloud shape: " + str(X_pcs.shape))
print("Y optlearner_params shape: " + str(Y_data[0].shape))
print("Y delta_d_loss shape: " + str(Y_data[1].shape))
print("Y optlearner_pc shape: " + str(Y_data[2].shape))
print("Y pointcloud_loss shape: " + str(Y_data[3].shape))
print("Y delta_d_hat_loss shape: " + str(Y_data[4].shape))
print("Y delta_d_hat_sin_loss shape: " + str(Y_data[5].shape))
print("Y smpl_loss shape: " + str(Y_data[6].shape))
print("Y delta_d shape: " + str(Y_data[7].shape))
print("Y delta_d_hat shape: " + str(Y_data[8].shape))
#print("Y delta_d_hat_NOGRAD shape: " + str(Y_data[9].shape))

#print("First X data params: " + str(X_data))

def update_weights_wrapper(batch_size):
    def update_weights(batch,logs):
        # Update a block of parameters
        PERIOD = 100
        BL_SIZE = batch_size // PERIOD
        #print("batch: " + str(batch))
        BL_INDEX = batch % PERIOD
        #print("BL_SIZE: " + str(BL_SIZE))
        #print("BL_INDEX: " + str(BL_INDEX))
        k = np.pi
        for param in trainable_params:
            layer = optlearner_model.get_layer(param)
            weights = np.array(layer.get_weights())
            #print("weights " + str(weights))
            #print('weights shape: '+str(weights.shape))
            #exit(1)
            #weights_new = [ (1-2*np.random.rand(weights[i].shape[0], weights[i].shape[1])) * k for i in range(len(weights))]
            weights_new = [ (1-2*np.random.rand(BL_SIZE, weights[i].shape[1])) * k for i in range(weights.shape[0])]
            #print("weights new " + str(weights_new))
            weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] = weights_new    # only update the required block of weights
            layer.set_weights(weights)
            #exit(1)
    return update_weights


# Callback for distractor unit
weight_cb_wrapper = update_weights_wrapper(data_samples)
weight_cb = LambdaCallback(on_epoch_end=lambda batch, logs: weight_cb_wrapper(batch, logs))


# Run the main loop
optlearner_model.fit(
    x=X_data,
    y=Y_data,
    epochs=100000,
    batch_size=32,
    shuffle=True,
    #steps_per_epoch=steps_per_epoch,
    #validation_data=val_gen,
    #validation_steps=validation_steps,
    callbacks=[epoch_pred_cb, model_save_checkpoint, weight_cb],
)


# Extract metadata
#from tensorflow.python.client import timeline
#tl = timeline.Timeline(run_metadata.step_stats)
#ctf = tl.generate_chrome_trace_format()
#with open('timeline.json', 'w') as f:
#    f.write(ctf)

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

