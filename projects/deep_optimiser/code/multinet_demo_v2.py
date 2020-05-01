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

#exp_dir = os.getcwd().replace("code", "")
#exp_dirs = [
#        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-23_08:04:34/",
#        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-22_20:14:34/",
#        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-22_20:15:12/"
#        ]
exp_dirs = [
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-27_21:28:23/",
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-27_21:29:45/",
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-27_21:30:23/",
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-27_21:30:53/",
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-27_21:31:47/",
        "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/NewDeepConv1DOptLearnerArchitecture_2020-04-27_21:32:22/"
]

all_setup_params = []
for exp_dir in exp_dirs:
    # Read in the configurations
    with open(exp_dir + "code/config.yaml", 'r') as f:
        try:
            setup_params = yaml.safe_load(f)
            #print(setup_params)
            all_setup_params.append(setup_params)
        except yaml.YAMLError as exc:
            print(exc)


# Gather shared parameters
setup_params = all_setup_params[0]

# Basic experimental setup
RESET_PERIOD = setup_params["BASIC"]["RESET_PERIOD"]
MODE = setup_params["BASIC"]["ROT_MODE"]
DISTRACTOR = setup_params["BASIC"]["DISTRACTOR"]
data_samples = setup_params["BASIC"]["NUM_SAMPLES"]
#num_cb_samples = setup_params["BASIC"]["NUM_CB_SAMPLES"]
num_cb_samples = 5

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
#LEARNING_RATES = setup_params["TEST"]["learning_rates"]
#num_test_samples = setup_params["TEST"]["num_test_samples"]
#JOINT_LEVELS = setup_params["TEST"]["joint_levels"]
LEARNING_RATES = [0.500]
num_test_samples = 5
JOINT_LEVELS = None

# Combine trainable joints of each network
trainable_params = []
all_params_trainable = []
param_ids = ["param_{:02d}".format(i) for i in range(85)]
for setup_params in all_setup_params:
    these_params = setup_params["PARAMS"]["TRAINABLE"]
    trainable_params += these_params
    params_trainable = {param: (param in these_params) for param in param_ids}
    all_params_trainable.append(params_trainable)

trainable_params = list(np.unique(trainable_params))


# Set the ID of this run
if args.run_id is not None:
    run_id = args.run_id
else:
    # Use the current date, time and model architecture as a run-id
    run_id = datetime.now().strftime("MultiNet_%Y-%m-%d_%H:%M:%S")
print("run_id: " + run_id)

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
print("gpu used:|" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")
#exit(1)

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import pickle
import copy

from callbacks import OptLearnerPredOnEpochEnd
from tools.model_helpers import initialise_emb_layers
from tools.test_helpers_v2 import setup_multinet_test_dir, setup_test_data, setup_embedding_weights, setup_test_model, initialise_pred_cbs
from optimisation_methods import learned_optimizer, regular_optimizer, multinet_optimizer


""" Test set-up """

#np.random.seed(10)
np.random.seed(11)

# Set-up experiment directory
models, logs_dirs, control_logs_dirs, test_vis_dirs, control_dirs = setup_multinet_test_dir(run_id, exp_dirs, LEARNING_RATES)

# Generate the data from the SMPL parameters
X_test, Y_test, X_cb, Y_cb, silh_cb, smpl, kin_tree, param_trainable, DISTRACTOR = setup_test_data(trainable_params, DISTRACTOR, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, JOINT_LEVELS, data_samples, num_test_samples=num_test_samples, num_cb_samples=num_cb_samples, MODE=MODE, LOAD_DATA_DIR=LOAD_DATA_DIR)

# Generate initial predictions
emb_initialiser, initial_weights = setup_embedding_weights(data_samples, X_test[1], DISTRACTOR, param_trainable)

optlearner_models = []
for i, model in enumerate(models):
    # Set-up the model
    optlearner_model = setup_test_model(model, ARCHITECTURE, data_samples, param_trainable, emb_initialiser, initial_weights, INPUT_TYPE, TRAIN_LR)
    optlearner_models.append(optlearner_model)

# Create the visualisation callbacks
epoch_pred_cbs = initialise_pred_cbs(logs_dirs, test_vis_dirs, smpl, X_cb, silh_cb, trainable_params, LEARNING_RATES)
epoch_pred_cbs_control = initialise_pred_cbs(control_logs_dirs, control_dirs, smpl, X_cb, silh_cb, trainable_params, LEARNING_RATES)


""" Test run """

if __name__ == "__main__":
    for lr_num, lr in enumerate(LEARNING_RATES):
        # Perform full optimisation for this learning rate
        print("\nTesting for lr '{:03f}'...".format(lr))
        multinet_optimizer(optlearner_models, X_test, all_params_trainable, epoch_pred_cbs[lr_num], num_samples=num_test_samples, epochs=TEST_EPOCHS, lr=lr, mode=MODE)

        # Reset the weights of the embedding layer
        print("Resetting predictions...")
        optlearner_model = initialise_emb_layers(optlearner_model, param_trainable, initial_weights)
    #regular_optimizer(optlearner_model, epochs=TEST_EPOCHS)


