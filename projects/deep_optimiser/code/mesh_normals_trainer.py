from __future__ import print_function

import os
import sys
import argparse
import json
from datetime import datetime
import yaml

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to configuration file")
parser.add_argument("--run_id", help="Identifier of this network pass")

args = parser.parse_args()

# Read in the configurations
if args.config is not None:
    with open(args.config, 'r') as f:
        setup_params = json.load(f)
else:
    with open("./config.yaml", 'r') as f:
        try:
            setup_params = yaml.safe_load(f)
            #print(setup_params)
        except yaml.YAMLError as exc:
            print(exc)

# Set the ID of this training pass
if args.run_id is not None:
    run_id = args.run_id
else:
    # Use the current date, time and model architecture as a run-id
    run_id = datetime.now().strftime("{}_%Y-%m-%d_%H:%M:%S".format(setup_params["MODEL"]["ARCHITECTURE"]))

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Set number of GPUs to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = setup_params["GENERAL"]["GPU_ID"]
print("gpu used:|" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")
#exit(1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

import keras
import keras.backend as K
from keras.callbacks import LambdaCallback, TensorBoard
from keras.models import Model
from keras.optimizers import Adam,SGD
import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from architectures.architecture_helpers import false_loss, no_loss, load_smpl_params, emb_init_weights
from smpl_np import SMPLModel
from posenet_maths_v5 import rotation_matrix_to_euler_angles
from smpl_np_rot_v6 import rodrigues
from euler_rodrigues_transform import rodrigues_to_euler

from render_mesh import Mesh
from callbacks import PredOnEpochEnd, OptLearnerPredOnEpochEnd, OptimisationCallback, OptLearnerLossGraphCallback
from generate_data import load_data
from training_helpers import update_weights_wrapper, offset_params, format_distractor_dict, architecture_output_array, architecture_inputs_and_outputs
from silhouette_generator import OptLearnerUpdateGenerator


""" Set-up """


# Create experiment directory
exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/" + str(run_id) + "/"
model_dir = exp_dir + "models/"
logs_dir = exp_dir + "logs/"
opt_logs_dir = logs_dir + "opt/"
train_vis_dir = exp_dir + "train_vis/"
test_vis_dir = exp_dir + "test_vis/"
opt_vis_dir = exp_dir + "opt_vis/"
code_dir = exp_dir + "code/"
tensorboard_logs_dir = logs_dir + "scalars/"
os.mkdir(exp_dir)
os.mkdir(model_dir)
os.mkdir(logs_dir)
os.mkdir(opt_logs_dir)
os.mkdir(train_vis_dir)
os.mkdir(test_vis_dir)
os.mkdir(opt_vis_dir)
os.mkdir(code_dir)
print("Experiment directory: \n" + str(exp_dir))
os.system("cp -r ./* " + str(code_dir))


""" Set-up and data generation """

np.random.seed(10)

# Parameter setup
trainable_params = setup_params["PARAMS"]["TRAINABLE"]
param_ids = ["param_{:02d}".format(i) for i in range(85)]

if trainable_params == "all_pose":
    not_trainable = [0, 1, 2]
    #not_trainable = []
    #trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
    trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 66]
    trainable_params = [param_ids[index] for index in trainable_params_indices]
elif trainable_params == "all_pose_and_global_rotation":
    not_trainable = [0, 2]
    #not_trainable = []
    #trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
    trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 66]
    trainable_params = [param_ids[index] for index in trainable_params_indices]

#trainable_params = ["param_14", "param_17", "param_59", "param_56"]
#trainable_params = ["param_01", "param_59", "param_56"]
#trainable_params = ["param_01", "param_59"]
#trainable_params = ["param_59", "param_56"]
#trainable_params = ["param_01"]
#trainable_params = ["param_56"]
#trainable_params = ["param_59"]

param_trainable = { param: (param in trainable_params) for param in param_ids }

# Basic experimental setup
RESET_PERIOD = setup_params["BASIC"]["RESET_PERIOD"]
MODEL_SAVE_PERIOD = setup_params["BASIC"]["MODEL_SAVE_PERIOD"]
PREDICTION_PERIOD = setup_params["BASIC"]["PREDICTION_PERIOD"]
OPTIMISATION_PERIOD = setup_params["BASIC"]["OPTIMISATION_PERIOD"]
MODE = setup_params["BASIC"]["ROT_MODE"]
DISTRACTOR = setup_params["BASIC"]["DISTRACTOR"]
data_samples = setup_params["BASIC"]["NUM_SAMPLES"]
DATA_LOAD_DIR = setup_params["DATA"]["TRAIN_DATA_DIR"]
POSE_OFFSET = setup_params["DATA"]["POSE_OFFSET"]
OFFSET_NT = setup_params["DATA"]["OFFSET_NT"]
PARAMS_TO_OFFSET = setup_params["DATA"]["PARAMS_TO_OFFSET"]
USE_GENERATOR = setup_params["DATA"]["USE_GENERATOR"]
ARCHITECTURE = setup_params["MODEL"]["ARCHITECTURE"]
BATCH_SIZE = setup_params["MODEL"]["BATCH_SIZE"]

# Format the distractor and offset dictionaries
DISTRACTOR = format_distractor_dict(DISTRACTOR, trainable_params)
if PARAMS_TO_OFFSET == "trainable_params":
    PARAMS_TO_OFFSET = trainable_params
elif PARAMS_TO_OFFSET == "all_pose_and_global_rotation":
    not_trainable = [0, 2]
    #not_trainable = []
    #trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
    trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 66]
    PARAMS_TO_OFFSET = [param_ids[index] for index in trainable_params_indices]
POSE_OFFSET = format_distractor_dict(POSE_OFFSET, PARAMS_TO_OFFSET)

# Generate the data from the SMPL parameters
smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
zero_params = np.zeros(shape=(85,))
zero_pc = smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85])
#print("zero_pc: " + str(zero_pc))

#print(PARAMS_TO_OFFSET)
#print(POSE_OFFSET)
if USE_GENERATOR:
    trainable_params_mask = [int(param_trainable[key]) for key in sorted(param_trainable.keys(), key=lambda x: int(x[6:8]))]
    update_generator = OptLearnerUpdateGenerator(data_samples, RESET_PERIOD, POSE_OFFSET, PARAMS_TO_OFFSET, ARCHITECTURE, batch_size=BATCH_SIZE, smpl=smpl, shuffle=True, save_path=train_vis_dir, trainable_params_mask=trainable_params_mask)
    X_data, Y_data = update_generator.yield_data()
    print("Y_data shapes: " + str([datum.shape for datum in Y_data]))
    X_indices = X_data[0]
    X_params = X_data[1]
    X_pcs = X_data[2]
    print("Generator length: " + str(len(update_generator)))
    print("X_indices shape: " + str(X_indices.shape))
    print("X_params shape: " + str(X_params.shape))
    print("X_pcs shape: " + str(X_pcs.shape))

else:
    # Generate/load and format the data
    X_indices = np.array([i for i in range(data_samples)])
    X_params = np.array([zero_params for i in range(data_samples)], dtype="float32")
    if DATA_LOAD_DIR is not None:
        X_params, X_pcs = load_data(DATA_LOAD_DIR, num_samples=data_samples, load_silhouettes=False)
    else:
        if not all(value == 0.0 for value in POSE_OFFSET.values()):
            X_params = offset_params(X_params, PARAMS_TO_OFFSET, POSE_OFFSET)
            X_pcs = np.array([smpl.set_params(beta=params[72:82], pose=params[0:72], trans=params[82:85]) for params in X_params])
        else:
            X_pcs = np.array([zero_pc for i in range(data_samples)], dtype="float32")

    if MODE == "EULER":
        # Convert from Rodrigues to Euler angles
        X_params = rodrigues_to_euler(X_params, smpl)

    X_data = [np.array(X_indices), np.array(X_params), np.array(X_pcs)]
    Y_data = architecture_output_array(ARCHITECTURE, data_samples, len(trainable_params))

    if ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture":
        trainable_params_mask = [int(param_trainable[key]) for key in sorted(param_trainable.keys(), key=lambda x: int(x[6:8]))]
        #print(trainable_params_mask)
        trainable_params_mask = np.tile(trainable_params_mask, (data_samples, 1))
        print("trainable_maks_shape: " + str(trainable_params_mask.shape))
        X_data += [trainable_params_mask]
        #exit(1)
#print(len(X_data))
#exit(1)

# Render silhouettes for the callback data
num_cb_samples = 5
cb_samples = np.linspace(0, data_samples, num_cb_samples, dtype=int)
cb_samples[-1] -= 1
X_cb = [entry[cb_samples] for entry in X_data]
Y_cb = [entry[cb_samples] for entry in Y_data]
cb_pcs = X_cb[2]
silh_cb = []
for pc in cb_pcs:
    silh = Mesh(pointcloud=pc).render_silhouette(show=False)
    silh_cb.append(silh)

# Validation data
#num_val_samples = 100
#val_samples = np.linspace(0, data_samples-1, num_val_samples, dtype=int)
#X_val = [entry[val_samples] for entry in X_data]
#Y_val = [entry[val_samples] for entry in Y_data]



""" Model set-up """

# Model setup parameters - ARCHITECTURE and BATCH_SIZE already defined above
EPOCHS = setup_params["MODEL"]["EPOCHS"]
INPUT_TYPE = setup_params["MODEL"]["INPUT_TYPE"]
learning_rate = setup_params["MODEL"]["LEARNING_RATE"]
DELTA_D_LOSS_WEIGHT = setup_params["MODEL"]["DELTA_D_LOSS_WEIGHT"]
PC_LOSS_WEIGHT = setup_params["MODEL"]["PC_LOSS_WEIGHT"]
DELTA_D_HAT_LOSS_WEIGHT = setup_params["MODEL"]["DELTA_D_HAT_LOSS_WEIGHT"]

# Embedding initialiser
#emb_initialiser = emb_init_weights(X_params, period=RESET_PERIOD, distractor=DISTRACTOR)   # if the initial distribution should match the stationary distribution
if OFFSET_NT:
    emb_initialiser = emb_init_weights(X_params, period=None, distractor=DISTRACTOR, pose_offset=POSE_OFFSET)
else:
    emb_initialiser = emb_init_weights(X_params, period=None, distractor=DISTRACTOR)

# Callback functions
# Create a model checkpoint after every few epochs
model_save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_dir + "model.{epoch:02d}-{delta_d_hat_mse_loss:.4f}.hdf5",
    monitor='loss', verbose=1, save_best_only=False, mode='auto',
    period=MODEL_SAVE_PERIOD, save_weights_only=True)

# Predict on sample params at the end of every few epochs
if USE_GENERATOR:
    epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=False, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples, train_gen=train_vis_dir)
else:
    epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=False, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples)
    #epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=True, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples)

# TensorBoard callback
#tensorboard_cb = TensorBoard(log_dir=tensorboard_logs_dir, histogram_freq=1, write_grads=True)

# Optimisation callback - perform an optimisation after every few epochs
#opt_lr = 0.5
#opt_epochs = 20
#opt_cb = OptimisationCallback(OPTIMISATION_PERIOD, opt_epochs, opt_lr, opt_logs_dir, smpl, X_data, train_inputs=X_cb, train_silh=silh_cb, pred_path=opt_vis_dir, period=1, trainable_params=trainable_params, visualise=False)


def print_summary_wrapper(path):
    def print_summary(s):
        with open(path + "model_summary.txt",'a') as f:
            f.write(s)
            print(s)
    return print_summary

# Build and compile the model
smpl_params, input_info, faces = load_smpl_params()
print("Optimiser architecture: " + str(ARCHITECTURE))
optlearner_inputs, optlearner_outputs = architecture_inputs_and_outputs(ARCHITECTURE, param_trainable, emb_initialiser, smpl_params, input_info, faces, data_samples, INPUT_TYPE)
print("optlearner inputs " +str(optlearner_inputs))
print("optlearner outputs "+str(optlearner_outputs))
optlearner_model = Model(inputs=optlearner_inputs, outputs=optlearner_outputs)
print_summary_fn = print_summary_wrapper(exp_dir)
optlearner_model.summary(print_fn=print_summary_fn)
#exit(1)

# Model options
#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata= tf.RunMetadata()

#learning_rate = 0.001
#learning_rate = 0.002
#learning_rate = 0.005
#optimizer= Adam(lr=learning_rate, decay=0.0001, clipvalue=1000.0)
optimizer= Adam(lr=learning_rate, decay=0.0)
#optimizer= SGD(lr=learning_rate, decay=0.0001, clipvalue=100.0)
#optimizer= SGD(lr=learning_rate, clipvalue=100.0)
#losses_to_show = ["delta_d_mse", "pc_mean_euc_dist", "delta_d_hat_mse"]
#losses_to_show = ["pc_mean_euc_dist"]
#losses_to_show = ["pc_mean_euc_dist", "delta_d_hat_mse"]
#losses_to_show = ["delta_d_mse", "diff_angle_norm", "delta_d_hat_mse"]
#loss_dict = {loss_name: false_loss for loss_name in losses_to_show}
#loss_weight_dict = {loss_name: (loss_name != "delta_d_hat_mse") for loss_name in losses_to_show}
#print("Loss dict: " + str(loss_dict))
#optlearner_model.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weight_dict)
#optlearner_model.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weight_dict, metrics={"delta_d_hat_sin_output": false_loss})
if INPUT_TYPE == "MESH_NORMALS":
    optlearner_loss = [no_loss,
            false_loss, # delta_d loss (L_smpl loss)
            no_loss,
            no_loss, # point cloud loss (L_xent)
            false_loss, # delta_d hat loss (L_delta_smpl)
            no_loss, # delta_d_hat sin metric
            no_loss, # this is the loss which updates smpl parameter inputs with predicted gradient
            no_loss, no_loss,
            false_loss # difference angle loss (L_xent)
            ]
    optlearner_loss_weights=[
            0.0,
            DELTA_D_LOSS_WEIGHT, # delta_d loss (L_smpl loss)
            0.0,
            0.0, # point cloud loss (L_xent)
            DELTA_D_HAT_LOSS_WEIGHT, # delta_d_hat loss (L_delta_smpl)
            0.0, # delta_d_hat sin metric - always set to 0
            0.0/learning_rate, # this is the loss which updates smpl parameter inputs with predicted gradient
            0.0, 0.0,
            PC_LOSS_WEIGHT, # difference angle loss (L_xent)
            ]

elif INPUT_TYPE == "3D_POINTS":
    optlearner_loss = [no_loss,
            false_loss, # delta_d loss (L_smpl loss)
            no_loss,
            false_loss, # point cloud loss (L_xent)
            false_loss, # delta_d hat loss (L_delta_smpl)
            no_loss, # delta_d_hat sin metric
            no_loss, # this is the loss which updates smpl parameter inputs with predicted gradient
            no_loss, no_loss,
            no_loss # difference angle loss (L_xent)
            ]
    optlearner_loss_weights=[
            0.0,
            DELTA_D_LOSS_WEIGHT, # delta_d loss (L_smpl loss)
            0.0,
            PC_LOSS_WEIGHT, # point cloud loss (L_xent)
            DELTA_D_HAT_LOSS_WEIGHT, # delta_d_hat loss (L_delta_smpl)
            0.0, # delta_d_hat sin metric - always set to 0
            0.0/learning_rate, # this is the loss which updates smpl parameter inputs with predicted gradient
            0.0, 0.0,
            0.0, # difference angle loss (L_xent)
            ]

if ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture" or ARCHITECTURE == "GroupedConv1DOptLearnerArchitecture":
    optlearner_loss += [false_loss]
    optlearner_loss_weights += [0.0]

if ARCHITECTURE == "RotConv1DOptLearnerArchitecture" or ARCHITECTURE == "ConditionalOptLearnerArchitecture":
    optlearner_loss += [false_loss, false_loss, false_loss, false_loss]
    optlearner_loss_weights += [0.0, 0.0, 0.0, 0.0]
    #optlearner_loss += [false_loss, false_loss, false_loss, false_loss, false_loss, false_loss]
    #optlearner_loss_weights += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

if ARCHITECTURE == "ProbCNNOptLearnerStaticArchitecture" or ARCHITECTURE == "GatedCNNOptLearnerArchitecture":
    optlearner_loss += [false_loss, false_loss, false_loss]
    optlearner_loss_weights += [0.0, 0.0, 0.0]

print("optlearner_loss length: " + str(len(optlearner_loss)))
print("optlearner_loss_weights length: " + str(len(optlearner_loss_weights)))
optlearner_model.compile(optimizer=optimizer, loss=optlearner_loss, loss_weights=optlearner_loss_weights,
                                            metrics={"delta_d_hat_sin_output": false_loss}
                                            #metrics={"delta_d_hat_sin_output": trainable_param_metric([int(param[6:8]) for param in trainable_params])}
                                            #options=run_options,
                                            #run_metadata=run_metadata
                                            )


# Callback for distractor unit
#weight_cb_wrapper = update_weights_wrapper(DISTRACTOR, data_samples, RESET_PERIOD, trainable_params, optlearner_model, X_params)
if USE_GENERATOR:
    weight_cb_wrapper = update_weights_wrapper(DISTRACTOR, data_samples, RESET_PERIOD, trainable_params, optlearner_model, generator=update_generator, offset_nt=OFFSET_NT, pose_offset=POSE_OFFSET)
else:
    weight_cb_wrapper = update_weights_wrapper(DISTRACTOR, data_samples, RESET_PERIOD, trainable_params, optlearner_model, offset_nt=OFFSET_NT, pose_offset=POSE_OFFSET)
weight_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: weight_cb_wrapper(epoch, logs))


# Callback for loss plotting during training
plotting_cb = OptLearnerLossGraphCallback(exp_dir, graphing_period=100)

""" Training loop"""

#print("X index shape: " + str(X_indices.shape))
#print("X parameters shape: " + str(X_params.shape))
#print("X point cloud shape: " + str(X_pcs.shape))
#print("Y optlearner_params shape: " + str(Y_data[0].shape))
#print("Y delta_d_loss shape: " + str(Y_data[1].shape))
#print("Y optlearner_pc shape: " + str(Y_data[2].shape))
#print("Y pointcloud_loss shape: " + str(Y_data[3].shape))
#print("Y delta_d_hat_loss shape: " + str(Y_data[4].shape))
#print("Y delta_d_hat_sin_loss shape: " + str(Y_data[5].shape))
#print("Y smpl_loss shape: " + str(Y_data[6].shape))
#print("Y delta_d shape: " + str(Y_data[7].shape))
#print("Y delta_d_hat shape: " + str(Y_data[8].shape))
#print("Y delta_d_hat_NOGRAD shape: " + str(Y_data[9].shape))


# Run the main loop
if USE_GENERATOR:
    optlearner_model.fit_generator(
            update_generator,
            steps_per_epoch=data_samples//BATCH_SIZE,
            epochs=EPOCHS,
            #max_queue_size=1,
            callbacks=[model_save_checkpoint, weight_cb, epoch_pred_cb, plotting_cb],
            #callbacks=[model_save_checkpoint, weight_cb, epoch_pred_cb],
            #callbacks=[weight_cb],
            shuffle=True,
            use_multiprocessing=False,
            workers=1
            )
else:
    optlearner_model.fit(
            x=X_data,
            y=Y_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            #steps_per_epoch=steps_per_epoch,
            #validation_data=(X_val, Y_val),
            #validation_steps=Y_val,
            #callbacks=[epoch_pred_cb, model_save_checkpoint, opt_cb, weight_cb],
            #callbacks=[epoch_pred_cb, model_save_checkpoint, opt_cb],
            #callbacks=[epoch_pred_cb, model_save_checkpoint, weight_cb],
            callbacks=[model_save_checkpoint, weight_cb, epoch_pred_cb, plotting_cb],
            #callbacks=[model_save_checkpoint, weight_cb, epoch_pred_cb],
            #callbacks=[epoch_pred_cb, tensorboard_cb],
            #callbacks=[epoch_pred_cb],
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

