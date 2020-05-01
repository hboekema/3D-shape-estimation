import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import cv2
import yaml
from datetime import datetime

# Parse the command-line arguments
parser = ArgumentParser()
parser.add_argument("--config", help="Path to configuration file")
parser.add_argument("--run_id", help="Identifier of this network pass")

args = parser.parse_args()

exp_dir = os.getcwd().replace("code", "")

# Read in the configurations
if args.config is not None:
    with open(args.config, 'r') as f:
        setup_params = json.load(f)
else:
    with open(exp_dir + "code/config.yaml", 'r') as f:
        try:
            setup_params = yaml.safe_load(f)
            #print(setup_params)
        except yaml.YAMLError as exc:
            print(exc)

# Load in the test configuration
# General setup
GPU_ID = setup_params["GENERAL"]["GPU_ID"]

# Basic experimental setup
RESET_PERIOD = setup_params["BASIC"]["RESET_PERIOD"]
MODEL_SAVE_PERIOD = setup_params["BASIC"]["MODEL_SAVE_PERIOD"]
PREDICTION_PERIOD = setup_params["BASIC"]["PREDICTION_PERIOD"]
OPIMISATION_PERIOD = setup_params["BASIC"]["OPTIMISATION_PERIOD"]
MODE = setup_params["BASIC"]["ROT_MODE"]
DISTRACTOR = setup_params["BASIC"]["DISTRACTOR"]
data_samples = setup_params["BASIC"]["NUM_SAMPLES"]
num_cb_samples = setup_params["BASIC"]["NUM_CB_SAMPLES"]

# Parameters setup
trainable_params = setup_params["PARAMS"]["TRAINABLE"]

# Data setup
LOAD_DATA_DIR = setup_params["DATA"]["TEST_DATA_DIR"]
POSE_OFFSET = setup_params["DATA"]["POSE_OFFSET"]
PARAMS_TO_OFFSET = setup_params["DATA"]["PARAMS_TO_OFFSET"]

# Model setup
ARCHITECTURE = setup_params["MODEL"]["ARCHITECTURE"]
INPUT_TYPE = setup_params["MODEL"]["INPUT_TYPE"]
TRAIN_LR = setup_params["MODEL"]["LEARNING_RATE"]

# Test setup
TEST_EPOCHS = setup_params["TEST"]["test_iterations"]
LEARNING_RATES = setup_params["TEST"]["learning_rates"]
num_test_samples = setup_params["TEST"]["num_test_samples"]
JOINT_LEVELS = setup_params["TEST"]["joint_levels"]


# Set the ID of this run (not currently used)
if args.run_id is not None:
    run_id = args.run_id
else:
    # Use the current date, time and model architecture as a run-id
    run_id = datetime.now().strftime("{}_%Y-%m-%d_%H:%M:%S".format(ARCHITECTURE))

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
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
print("gpu used:|" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")
#exit(1)

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from keras.models import Model
from keras.optimizers import Adam, SGD
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import pickle
import copy

from callbacks import OptLearnerPredOnEpochEnd
from architectures.architecture_helpers import false_loss
from smpl_np import SMPLModel
from render_mesh import Mesh
from tools.data_helpers import format_offsetable_params, offset_params, format_distractor_dict, architecture_output_array, gather_input_data, gather_cb_data, format_joint_levels
from tools.test_helpers import get_exp_models, create_base_subdir, create_subdir, initialise_pred_cbs
from tools.model_helpers import emb_init_weights_np, construct_optlearner_model, freeze_layers, initialise_emb_layers, gather_optlearner_losses
from optimisation_methods import learned_optimizer, regular_optimizer


""" Test set-up """

#np.random.seed(10)
np.random.seed(11)

# Gather experiment directory details
model_name = get_exp_models(exp_dir)
if JOINT_LEVELS is None:
    save_suffix = ""
else:
    save_suffix = "_conditional"
#save_suffix = "_non-zero_pose"

# Create base testing directory
model, logs_dir, control_logs_dir, test_vis_dir, control_dir = create_base_subdir(exp_dir, model_name, save_suffix)

# Create subdirectories for the learning rates
logs_dirs = []
control_logs_dirs = []
test_vis_dirs = []
control_dirs = []
for lr in LEARNING_RATES:
    sub_dir = "lr_{:02f}/".format(lr)
    logs_dir, control_logs_dir, test_vis_dir, control_dir = create_subdir(exp_dir, model_name, sub_dir, save_suffix)
    logs_dirs.append(logs_dir)
    control_logs_dirs.append(control_logs_dir)
    test_vis_dirs.append(test_vis_dir)
    control_dirs.append(control_dir)


# Generate the data from the SMPL parameters
# Gather trainable params
param_ids = ["param_{:02d}".format(i) for i in range(85)]
trainable_params = format_offsetable_params(trainable_params)
param_trainable = { param: (param in trainable_params) for param in param_ids }
DISTRACTOR = format_distractor_dict(DISTRACTOR, trainable_params)

# Gather offsetable params
if PARAMS_TO_OFFSET == "trainable_params":
    PARAMS_TO_OFFSET = trainable_params
PARAMS_TO_OFFSET = format_offsetable_params(PARAMS_TO_OFFSET)
POSE_OFFSET = format_distractor_dict(POSE_OFFSET, PARAMS_TO_OFFSET)

# Define the kinematic tree
kin_tree = format_joint_levels(JOINT_LEVELS)

# Generate the data from the SMPL parameters
print("loading SMPL...")
smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

# Generate and format the data
print("Gather input data...")
X_test, Y_test = gather_input_data(data_samples, smpl, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, param_trainable, num_test_samples=num_test_samples, MODE=MODE, LOAD_DATA_DIR=LOAD_DATA_DIR)

# Render silhouettes for the callback data
X_cb, Y_cb, silh_cb = gather_cb_data(X_test, Y_test, data_samples, num_cb_samples, where="front")

# Generate initial embedding layer weights
initial_weights = np.zeros((data_samples, 85))
emb_initialiser = emb_init_weights_np(X_test[1], distractor=DISTRACTOR)
for param_name, trainable in param_trainable.items():
    param_number = int(param_name[6:8])
    emb_init_ = emb_initialiser(param=param_number, offset=trainable)
    initial_weights[:, param_number] = emb_init_(shape=(data_samples,))

# Retrieve model architecture and load weights
optlearner_model = construct_optlearner_model(ARCHITECTURE, param_trainable, emb_initialiser, data_samples, INPUT_TYPE)
optlearner_model.load_weights(model)

# Freeze all layers except for the required embedding layers
optlearner_model = freeze_layers(optlearner_model, param_trainable)

# Set the weights of the embedding layers
optlearner_model = initialise_emb_layers(optlearner_model, param_trainable, initial_weights)

# Compile the model
optimizer = Adam(lr=TRAIN_LR, decay=0.0)
#optimizer = SGD(lr=TRAIN_LR, momentum=0.0, nesterov=False)
optlearner_loss, optlearner_loss_weights = gather_optlearner_losses(INPUT_TYPE, ARCHITECTURE)
optlearner_model.compile(
        optimizer=optimizer,
        loss=optlearner_loss,
        loss_weights=optlearner_loss_weights,
        metrics={"delta_d_hat_sin_output": false_loss},
        #options=run_options,
        #run_metadata=run_metadata,
        )

# Print model summary
optlearner_model.summary()


# Visualisation callbacks
epoch_pred_cbs = initialise_pred_cbs(logs_dirs, test_vis_dirs, smpl, X_cb, silh_cb, trainable_params, LEARNING_RATES)
epoch_pred_cbs_control = initialise_pred_cbs(control_logs_dirs, control_dirs, smpl, X_cb, silh_cb, trainable_params, LEARNING_RATES)


if __name__ == "__main__":
    for lr_num, lr in enumerate(LEARNING_RATES):
        print("\nTesting for lr '{:02f}'...".format(lr))
        learned_optimizer(optlearner_model, X_test, param_trainable, epoch_pred_cbs[lr_num], num_samples=num_test_samples, epochs=TEST_EPOCHS, lr=lr, mode=MODE, kinematic_levels=kin_tree)
        # Reset the weights of the embedding layer
        for layer_name, trainable in param_trainable.items():
            param_number = int(layer_name[6:8])
            emb_layer = optlearner_model.get_layer(layer_name)
            layer_init_weights = initial_weights[:, param_number]
            emb_layer.set_weights(layer_init_weights.reshape(1, -1, 1))
    #regular_optimizer(optlearner_model, epochs=TEST_EPOCHS)


