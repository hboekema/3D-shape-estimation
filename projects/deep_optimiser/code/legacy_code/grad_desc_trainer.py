from __future__ import print_function

import os
import sys
import argparse
import json
from datetime import datetime
import keras
import keras.backend as K
from keras.callbacks import LambdaCallback, TensorBoard
from keras.models import Model
from keras.optimizers import Adam,SGD
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import pickle

sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
#from optlearner import OptLearnerCombinedStaticModArchitecture, OptLearnerMeshNormalStaticArchitecture, OptLearnerMeshNormalStaticModArchitecture, OptLearnerBothNormalsStaticSinArchitecture, OptLearnerMeshNormalStaticSinArchitecture, false_loss, no_loss, load_smpl_params, emb_init_weights
from architectures.OptLearnerCombinedStaticModArchitecture import OptLearnerCombinedStaticModArchitecture
from architectures.OptLearnerMeshNormalStaticArchitecture import OptLearnerMeshNormalStaticArchitecture
from architectures.OptLearnerMeshNormalStaticModArchitecture import OptLearnerMeshNormalStaticModArchitecture
from architectures.FullOptLearnerStaticArchitecture import FullOptLearnerStaticArchitecture
from architectures.Conv1DFullOptLearnerStaticArchitecture import Conv1DFullOptLearnerStaticArchitecture
from architectures.GradDescArchitecture import GradDescArchitecture
#from architectures.architecture_helpers import false_loss, no_loss, trainable_param_metric, load_smpl_params, emb_init_weights
from architectures.architecture_helpers import false_loss, no_loss, load_smpl_params, emb_init_weights
from smpl_np import SMPLModel
from posenet_maths_v5 import rotation_matrix_to_euler_angles
from smpl_np_rot_v6 import rodrigues, print_mesh
from euler_rodrigues_transform import rodrigues_to_euler

from render_mesh import Mesh
from callbacks import PredOnEpochEnd, OptLearnerPredOnEpochEnd, OptimisationCallback
from generate_data import load_data
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
    run_id = datetime.now().strftime("grad_desc_%Y-%m-%d_%H:%M:%S")

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


# Set number of GPUs to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("gpu used:|" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")
#exit(1)

# Ensure that TF2.0 is not used
#tf.disable_v2_behavior()
#tf.enable_eager_execution()

# Set Keras format
#tf.keras.backend.set_image_data_format(params["ENV"]["CHANNEL_FORMAT"])

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
#not_trainable = [0, 2]
not_trainable = []
#trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
trainable_params_indices = [index for index in range(85) if index not in not_trainable]
#trainable_params = [param_ids[index] for index in trainable_params_indices]

#trainable_params = ["param_00", "param_01", "param_02", "param_56", "param_57", "param_58", "param_59", "param_60", "param_61"]
trainable_params = ["param_01", "param_59", "param_56"]
#trainable_params = ["param_01", "param_59"]
#trainable_params = ["param_14", "param_17", "param_59", "param_56"]
#trainable_params = ["param_59", "param_56"]
#trainable_params = ["param_01"]
#trainable_params = ["param_56"]
#trainable_params = ["param_59"]
#trainable_params = []

param_trainable = { param: (param in trainable_params) for param in param_ids }
# Generate the data from the SMPL parameters
smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
zero_params = np.zeros(shape=(85,))
zero_pc = smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85])
#print("zero_pc: " + str(zero_pc))
#base_params = 0.2 * (np.random.rand(85) - 0.5)
#base_pc = smpl.set_params(beta=base_params[72:82], pose=base_params[0:72].reshape((24,3)), trans=base_params[82:85])

# Kinematic tree
kinematic_tree = {"param_01": ["param_56", "param_59"]}
def get_tree_levels(tree):
    levels = {}

    nonroots = {i for node in tree.values() for i in node}
    roots = [k for k in tree if k not in nonroots]
    generation = 0
    while roots:
        levels[generation] = roots
        generation += 1
        roots = [c for k in roots for c in tree.get(k,())]
    return levels
kinematic_tree_levels = get_tree_levels(kinematic_tree)

# Parameters
dtype = "float64"
#RESET_PERIOD = 1000
#RESET_PERIOD = 200
#RESET_PERIOD = 50
#RESET_PERIOD = 20
#RESET_PERIOD = 10
RESET_PERIOD = None
#MODEL_SAVE_PERIOD = 1
#MODEL_SAVE_PERIOD = 10
MODEL_SAVE_PERIOD = 50
#MODEL_SAVE_PERIOD = 500000
PREDICTION_PERIOD = 1
#PREDICTION_PERIOD = 10
#OPTIMISATION_PERIOD = 5
#OPTIMISATION_PERIOD = 10
OPTIMISATION_PERIOD = 50
MODE = "RODRIGUES"
#MODE = "EULER"
#DISTRACTOR = np.pi
#DISTRACTOR = 1.0
DISTRACTOR = 0.1
#DISTRACTOR = 0.3
#DISTRACTOR = 0.0
DATA_DIR = "/data/cvfs/hjhb2/data/artificial/"

# Data collection
#data_samples = 10000
#data_samples = 1000
data_samples = 100
#data_samples = 10
#data_samples = 1

# Load data
X_indices = np.array([i for i in range(data_samples)])
#X_params, X_pcs = load_data(DATA_DIR + "train/", data_samples, load_silhouettes=False)
#X_params = 0.2 * 2*(np.random.rand(data_samples, 85) - 0.5)
X_params = np.array([zero_params for i in range(data_samples)], dtype=dtype)

def trainable_param_dist(X_params, trainable_params, k=np.pi):
    trainable_params_int = [int(param[6:8]) for param in trainable_params]
    X_params[:, trainable_params_int] = 1.0
    #X_params[:, trainable_params_int] = 2 * k * (np.random.rand(data_samples, len(trainable_params_int)) - 0.5)

    return X_params

#X_params = trainable_param_dist(X_params, trainable_params, DISTRACTOR)
#X_pcs = np.array([zero_pc for i in range(data_samples)], dtype=dtype)
#smpl_params, input_info, faces = load_smpl_params()
#print_mesh("zero_mesh.obj", X_pcs[10], faces)

#X_pcs = np.array([smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85]) for i in range(data_samples)])
#allclose = [np.allclose(X_pc, zero_pc) for X_pc in X_pcs]
#print(allclose)
#exit(1)

#X_params = np.array([base_params for i in range(data_samples)], dtype=dtype)
#X_pcs = np.array([base_pc for i in range(data_samples)], dtype=dtype)
X_pcs = []
for params in X_params:
     pc = smpl.set_params(beta=params[72:82], pose=params[0:72], trans=params[82:85])
     X_pcs.append(pc)
X_pcs = np.array(X_pcs)

if MODE == "EULER":
    # Convert from Rodrigues to Euler angles
    X_params = rodrigues_to_euler(X_params, smpl)

X_data = [np.array(X_indices), np.array(X_params), np.array(X_pcs)]
#Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,))]
Y_data = [np.zeros((data_samples,))]
#Y_data = [np.zeros((data_samples, 6890))]
#Y_data = [np.zeros((data_samples, 6890, 3))]

# Render silhouettes for the callback data
#num_samples = 5
#cb_indices = X_indices[:num_samples]
#cb_params = X_params[:num_samples]
#cb_pcs = X_pcs[:num_samples]
#cb_samples = [0, 2000, 4000, 8000, 9999]
#cb_samples = [0, 200, 400, 800, 999]
cb_samples = [0, 20, 40, 80, 99]
cb_indices = X_indices[cb_samples]
cb_params = X_params[cb_samples]
cb_pcs = X_pcs[cb_samples]
X_cb = [np.array(cb_indices), np.array(cb_params), np.array(cb_pcs)]
Y_cb = [entry[cb_indices] for entry in Y_data]
silh_cb = []
for pc in cb_pcs:
    silh = Mesh(pointcloud=pc).render_silhouette(show=False)
    silh_cb.append(silh)

# Validation data
num_val_samples = 100
val_samples = np.linspace(0, data_samples-1, num_val_samples, dtype=int)
X_val = [entry[val_samples] for entry in X_data]
Y_val = [entry[val_samples] for entry in Y_data]


""" Model set-up """

# Embedding initialiser
emb_initialiser = emb_init_weights(X_params, period=RESET_PERIOD, distractor=DISTRACTOR)

# Callback functions
# Create a model checkpoint after every few epochs
model_save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_dir + "model.{epoch:02d}-{delta_d_hat_mse_loss:.4f}.hdf5",
    monitor='loss', verbose=1, save_best_only=False, mode='auto',
    period=MODEL_SAVE_PERIOD, save_weights_only=True)

# Predict on sample params at the end of every few epochs
#epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=False, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples)
epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=True, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples)

# TensorBoard callback
tensorboard_cb = TensorBoard(log_dir=tensorboard_logs_dir, histogram_freq=1, write_grads=True)

# Perform an optimisation after every few epochs
opt_lr = 0.5
opt_epochs = 20
opt_cb = OptimisationCallback(OPTIMISATION_PERIOD, opt_epochs, opt_lr, opt_logs_dir, smpl, X_data, train_inputs=X_cb, train_silh=silh_cb, pred_path=opt_vis_dir, period=1, trainable_params=trainable_params, visualise=False)


# Build and compile the model
smpl_params, input_info, faces = load_smpl_params()
#optlearner_inputs, optlearner_outputs = OptLearnerMeshNormalStaticModArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples)
#optlearner_inputs, optlearner_outputs = FullOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples)
#optlearner_inputs, optlearner_outputs = Conv1DFullOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples)
optlearner_inputs, optlearner_outputs = GradDescArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples)
print("optlearner inputs " +str(optlearner_inputs))
print("optlearner outputs "+str(optlearner_outputs))
optlearner_model = Model(inputs=optlearner_inputs, outputs=optlearner_outputs)
optlearner_model.summary()

# Model options
#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata= tf.RunMetadata()

#learning_rate = 400
#learning_rate = 20
#learning_rate = 10
#learning_rate = 4
#learning_rate = 1
#learning_rate = 0.5
#learning_rate = 0.1
learning_rate = 0.01
#learning_rate = 0.005 #0.001
#learning_rate = 0.001
#learning_rate = 0.02 #0.001
#optimizer= Adam(lr=learning_rate, decay=0.0001, clipvalue=1000.0)
optimizer= Adam(lr=learning_rate, decay=0.0)
#optimizer= SGD(lr=learning_rate, decay=0.0001, clipvalue=100.0)
#optimizer= SGD(lr=learning_rate, clipvalue=100.0)
#losses_to_show = ["delta_d_mse", "pc_mean_euc_dist", "delta_d_hat_mse"]
#losses_to_show = ["delta_d_mse"]
losses_to_show = ["pc_mean_euc_dist"]
#losses_to_show = ["pc_mean_euc_dist", "delta_d_hat_mse"]
#losses_to_show = ["delta_d_mse", "diff_angle_norm", "delta_d_hat_mse"]
loss_dict = {loss_name: false_loss for loss_name in losses_to_show}
loss_weight_dict = {loss_name: (loss_name != "delta_d_hat_mse") for loss_name in losses_to_show}
print("Loss dict: " + str(loss_dict))
optlearner_model.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weight_dict)
#optlearner_model.compile(optimizer=optimizer, loss=loss_dict, loss_weights=loss_weight_dict, metrics={"delta_d_hat_sin_output": false_loss})
#optlearner_model.compile(optimizer=optimizer, loss=[no_loss,
#                                                   no_loss, # delta_d loss (L_smpl loss)
#                                                   no_loss,
#                                                   false_loss, # point cloud loss (L_xent)
#                                                   false_loss, # delta_d hat loss (L_delta_smpl)
#                                                   no_loss, # delta_d_hat sin metric
#						   no_loss, # this is the loss which updates smpl parameter inputs with predicted gradient
#						   no_loss, no_loss,
#                                                   no_loss # difference angle loss (L_xent)
#                                                   #no_loss, no_loss, no_loss
#                                                   ],
#                                            loss_weights=[0.0,
#                                                        0.0*1, # delta_d loss (L_smpl loss)
#                                                        0.0,
#                                                        1.0*1, # point cloud loss (L_xent)
#                                                        0.0*1, # delta_d_hat loss (L_delta_smpl)
#                                                        0.0, # delta_d_hat sin metric - always set to 0
#                                                        0.0/learning_rate, # this is the loss which updates smpl parameter inputs with predicted gradient
#                                                        0.0, 0.0,
#                                                        0.0*1, # difference angle loss (L_xent)
#                                                        #0.0, 0.0, 0.0
#                                                        ],
#                                            metrics={"delta_d_hat_sin_output": false_loss}
#                                            #metrics={"delta_d_hat_sin_output": trainable_param_metric([int(param[6:8]) for param in trainable_params])}
#                                            #options=run_options,
#                                            #run_metadata=run_metadata
#                                            )

# Initialise the embedding layer weights
#initial_weights = emb_init(shape=(1, 1000, 85))
#emb_layer = optlearner_model.get_layer("parameter_embedding")
#emb_layer.set_weights(initial_weights)

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

#print("First X data params: " + str(X_data))

def update_weights_wrapper(DISTRACTOR, batch_size, PERIOD, gt_params=None):
    def update_weights(batch,logs):
        # May need to verify that this works correctly - visually it looks okay but something might be wrong
        # Update a block of parameters
        BL_SIZE = batch_size // PERIOD
        #print("batch: " + str(batch))
        BL_INDEX = batch % PERIOD
        #print("BL_SIZE: " + str(BL_SIZE))
        #print("BL_INDEX: " + str(BL_INDEX))
        k = DISTRACTOR
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

    def update_weights_artificially(batch, logs):
        k = DISTRACTOR
        for param in trainable_params:
            gt_weights = gt_params[:, int(param[6:8])]
            layer = optlearner_model.get_layer(param)
            weights = np.array(layer.get_weights())
            #print("weights " + str(weights))
            #print('weights shape: '+str(weights.shape))
            #exit(1)

            # Compute the offset from the ground truth value
            offset_value = np.random.uniform(low=-k, high=k, size=(batch_size, 1))
            #offset_value = K.random_uniform(shape=[batch_size], minval=-k, maxval=k, dtype=dtype)
            block_size = batch_size // PERIOD
            means = [np.sqrt(float((i + 1 + batch) % PERIOD)/PERIOD) for i in range(PERIOD)]
            np.random.shuffle(means)
            factors = np.concatenate([np.random.normal(loc=means[i], scale=0.01, size=(block_size, 1)) for i in range(PERIOD)])
            offset_value *= factors
            #print("offset_value shape: " + str(offset_value.shape))
            new_weights = gt_weights.reshape(offset_value.shape) + offset_value
            new_weights = new_weights.reshape(weights.shape)
            #print("new_weights shape: " + str(new_weights.shape))
            #factors = Concatenate()([K.random_normal(shape=[block_size], mean=means[i], stddev=0.01) for i in range(PERIOD)])
            #offset_value = Multiply()([offset_value, factors])
            #weights = Add()([gt_weights, offset_value])

            layer.set_weights(new_weights)
            #exit(1)

    def update_weights_kinematically(batch, logs):
        k = DISTRACTOR
        tree_levels = [level for level in kinematic_tree_levels for param in trainable_params if param in level]
        base_block_size = batch_size // len(tree_levels)
        remainder_size = batch_size % len(tree_levels)
        trainable_params_sizes = {param: base_block_size for param in trainable_params}
        trainable_params_sizes[trainable_params[0]] += remainder_size

        new_weights = {}
        for param, block_size in trainable_params_sizes.items():

            gt_weights = gt_params[:, int(param[6:8])]
            layer = optlearner_model.get_layer(param)
            weights = np.array(layer.get_weights())
            #print("weights " + str(weights))
            #print('weights shape: '+str(weights.shape))
            #exit(1)

            # Compute the offset from the ground truth value
            offset_value = np.random.uniform(low=-k, high=k, size=(batch_size, 1))
            #offset_value = K.random_uniform(shape=[batch_size], minval=-k, maxval=k, dtype=dtype)
            block_size = batch_size // PERIOD
            means = [np.sqrt(float((i + 1 + batch) % PERIOD)/PERIOD) for i in range(PERIOD)]
            np.random.shuffle(means)
            factors = np.concatenate([np.random.normal(loc=means[i], scale=0.01, size=(block_size, 1)) for i in range(PERIOD)])
            offset_value *= factors
            #print("offset_value shape: " + str(offset_value.shape))
            new_weights = gt_weights.reshape(offset_value.shape) + offset_value
            new_weights = new_weights.reshape(weights.shape)
            #print("new_weights shape: " + str(new_weights.shape))
            #factors = Concatenate()([K.random_normal(shape=[block_size], mean=means[i], stddev=0.01) for i in range(PERIOD)])
            #offset_value = Multiply()([offset_value, factors])
            #weights = Add()([gt_weights, offset_value])

            layer.set_weights(new_weights)
            #exit(1)

    if gt_params is None:
        return update_weights
    else:
        return update_weights_artificially


# Callback for distractor unit
#weight_cb_wrapper = update_weights_wrapper(DISTRACTOR, data_samples, RESET_PERIOD, X_params)
weight_cb_wrapper = update_weights_wrapper(DISTRACTOR, data_samples, RESET_PERIOD)
weight_cb = LambdaCallback(on_epoch_end=lambda batch, logs: weight_cb_wrapper(batch, logs))

class loss_calculation_callback(keras.callbacks.Callback):
    def __init__(self, data, model=None):
        #self.set_model(model)
        self.data = data

    def on_epoch_begin(self, epoch, logs=None):
        import cv2
        outputs = self.model.predict(self.data)
        #print(outputs)

        gt_params = self.data[1]
        pred_params = outputs[0]
        print("GT and predicted parameters are equal: " + str(np.allclose(gt_params, pred_params)))

        #print(gt_params)
        #print(pred_params)
        #exit(1)

        gt_pc = self.data[2]
        #print(gt_pc)
        pred_pc = outputs[2]
        right_pred = Mesh(pointcloud=pred_pc[0]).render_silhouette(show=False)
        cv2.imwrite("right_pred.png", right_pred.astype("uint8"))
        wrong_pred = Mesh(pointcloud=pred_pc[10]).render_silhouette(show=False)
        cv2.imwrite("wrong_pred.png", wrong_pred.astype("uint8"))
        print_mesh("wrong_mesh.obj", pred_pc[10], faces)
        print_mesh("right_mesh.obj", pred_pc[0], faces)
        #print(pred_pc)
        print("GT and predicted point clouds are equal: " + str(np.allclose(gt_pc, pred_pc)))
        close = np.array([np.int(np.allclose(gt_pc[i], pred_pc[i])) for i in range(gt_pc.shape[0])])
        #print([np.allclose(gt_pc[i], pred_pc[i]) for i in range(gt_pc.shape[0])])
        not_close = np.array([not value for value in close])
        close_sum = np.sum(close)
        #print("Num close pc: " + str(close_sum))
        not_close_gt = gt_pc[not_close]
        not_close_pred = pred_pc[not_close]
        #print(not_close_pred[0])
        #exit(1)
        diff_not_close = not_close_gt - not_close_pred
        import pandas as pd
        diff_df = pd.DataFrame(diff_not_close[10])
        diff_df.to_csv("diff.csv")
        dist_not_close = np.sum(np.square(diff_not_close), axis=-1)
        dist_df = pd.DataFrame(dist_not_close[10])
        dist_df.to_csv("dist.csv")
        mean_not_close = np.mean(dist_not_close, axis=1)
        #print(mean_not_close)

        gt_example = not_close_gt[10]
        pred_example = not_close_pred[10]

        gt_silh = Mesh(pointcloud=gt_example).render_silhouette(show=False)
        pred_silh = Mesh(pointcloud=pred_example).render_silhouette(show=False)
        diff_silh = (gt_silh != pred_silh)*255
        all_silh = np.concatenate([gt_silh, pred_silh, diff_silh])
        cv2.imwrite("silh_comp.png", all_silh.astype("uint8"))


        #exit(1)
        euc_dist = np.square(np.subtract(gt_pc, pred_pc))
        #print(euc_dist[0])
        euc_dist_summed = np.sum(euc_dist, axis=-1)
        #print(euc_dist_summed[0])
        actual_loss = np.mean(euc_dist_summed, axis=1)
        #print(actual_loss)
        actual_loss_sum = np.sum(actual_loss, axis=0)
        #print(actual_loss_sum)

        mean_loss = np.mean(actual_loss)
        print("Calculated mean loss: {}".format(mean_loss))
        print("Model mean loss: {}".format(np.mean(outputs[3])))
        #exit(1)


loss_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: print("Average loss after epoch " + str(epoch) + ": " + str(logs["loss"])))
#loss_cb = loss_calculation_callback(X_data)

#batch_size = 1
#batch_size = 5
batch_size = 100
evaluation = optlearner_model.evaluate(X_data, Y_data, batch_size=batch_size)
print("Initial loss: " + str(evaluation[0]))
#exit(1)

# Run the main loop
optlearner_model.fit(
    x=X_data,
    y=Y_data,
    epochs=100000,
    batch_size=batch_size,
    #shuffle=True,
    shuffle=False,
    #steps_per_epoch=steps_per_epoch,
    #validation_data=(X_val, Y_val),
    #validation_steps=Y_val,
    #callbacks=[epoch_pred_cb, model_save_checkpoint, opt_cb, weight_cb],
    #callbacks=[epoch_pred_cb, model_save_checkpoint, opt_cb],
    #callbacks=[epoch_pred_cb, model_save_checkpoint, weight_cb],
    #callbacks=[epoch_pred_cb, tensorboard_cb],
    #callbacks=[epoch_pred_cb],
    #callbacks=[tensorboard_cb],
    callbacks=[loss_cb],
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

