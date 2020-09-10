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
    config_path = args.config
else:
    config_path = "./config.yaml"

with open(config_path, 'r') as f:
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
TRAIN_DISTRACTOR = setup_params["BASIC"]["DISTRACTOR"]
data_samples = setup_params["BASIC"]["NUM_SAMPLES"]
num_cb_samples = setup_params["BASIC"]["NUM_CB_SAMPLES"]

# Parameters setup
trainable_params = setup_params["PARAMS"]["TRAINABLE"]

# Data setup
LOAD_DATA_DIR = setup_params["DATA"]["TEST_DATA_DIR"]
OFFSET_NT = setup_params["DATA"]["OFFSET_NT"]
RESET_PRED_TO_ZERO = setup_params["DATA"]["RESET_PRED_TO_ZERO"]
TRAIN_OFFSET = setup_params["DATA"]["POSE_OFFSET"]
PARAMS_TO_OFFSET = setup_params["DATA"]["PARAMS_TO_OFFSET"]
TRAIN_DIST = setup_params["DATA"]["DIST"]
INCLUDE_SHAPE = setup_params["DATA"]["INCLUDE_SHAPE"]

# Model setup
ARCHITECTURE = setup_params["MODEL"]["ARCHITECTURE"]
INPUT_TYPE = setup_params["MODEL"]["INPUT_TYPE"]
TRAIN_LR = setup_params["MODEL"]["LEARNING_RATE"]

# Test setup
TEST_EPOCHS = setup_params["TEST"]["test_iterations"]
LEARNING_RATES = setup_params["TEST"]["learning_rates"]
normals_lr_weights = setup_params["TEST"]["normals_lr_weights"]
points_lr_weights = setup_params["TEST"]["points_lr_weights"]
assert (len(LEARNING_RATES) == len(normals_lr_weights) and len(LEARNING_RATES) == len(points_lr_weights)), "loss weight lists must have equal length"

num_test_samples = setup_params["TEST"]["num_test_samples"]
OPTIMIZER = setup_params["TEST"]["OPTIMIZER"]
JOINT_LEVELS = setup_params["TEST"]["joint_levels"]
update_weight = LEARNING_RATES[0]
TEST_DIST = setup_params["TEST"]["TEST_DIST"]
TEST_DISTRACTOR = setup_params["TEST"]["TEST_DISTRACTOR"]
TEST_OFFSET = setup_params["TEST"]["TEST_POSE_OFFSET"]
TEST_BATCH_SIZE = setup_params["TEST"]["BATCH_SIZE"]

# Multiple test scenarios need to be specified
distributions = [TEST_DIST, TEST_DIST, TRAIN_DIST]
distractors = [TEST_DISTRACTOR, TEST_DISTRACTOR, TRAIN_DISTRACTOR]
pose_offsets = [TEST_OFFSET, TEST_OFFSET, TRAIN_OFFSET]
preds_init = [False, True, RESET_PRED_TO_ZERO]
are_train_dist = [False, False, True]

test_distributions = zip(distributions, distractors, pose_offsets, preds_init, are_train_dist)
#test_distributions = [test_distributions[-1]]
print("test scenarios: " + str(test_distributions))


# Set the ID of this run (not currently used)
if args.run_id is not None:
    run_id = args.run_id
else:
    # Use the current date, time and model architecture as a run-id
    run_id = datetime.now().strftime("{}_%Y-%m-%d_%H:%M:%S".format(ARCHITECTURE))

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.backend import clear_session
print("TF version: " + str(tf.__version__))

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

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import pickle
import copy

from tools.model_helpers_v2 import initialise_emb_layers
from tools.test_helpers_v4 import get_exp_models, create_test_dir_for_model, setup_test_data, setup_embedding_weights, setup_test_model, initialise_pred_cbs, gen_save_suffix, get_test_type
from optimisation_methods import learned_optimizer, regular_optimizer, learned_optimizer_gradient_updates
from metrics_plots import plot_metrics


""" Test set-up """


if __name__ == "__main__":
    #np.random.seed(10)
    np.random.seed(11)

    # Collect relevant models
    model_names = get_exp_models(exp_dir)
    for model_name in model_names:
        for test_scenario in test_distributions:
            DIST, DISTRACTOR, POSE_OFFSET, RESET_PRED_TO_ZERO, is_train_dist = test_scenario

            # Set-up experiment directory
            save_suffix = gen_save_suffix(JOINT_LEVELS, DIST, RESET_PRED_TO_ZERO, is_train_dist)
            model, logs_dirs, control_logs_dirs, test_vis_dirs, control_dirs, test_vis_model_dir = create_test_dir_for_model(exp_dir, model_name, save_suffix, LEARNING_RATES, points_lr_weights=points_lr_weights, normals_lr_weights=normals_lr_weights, OPTIMIZER=OPTIMIZER)

            # Generate the data from the SMPL parameters
            X_test, Y_test, X_cb, Y_cb, silh_cb, smpl, kin_tree, trainable_params, param_trainable, DISTRACTOR, POSE_OFFSET = setup_test_data(trainable_params, DISTRACTOR, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, JOINT_LEVELS, data_samples, num_test_samples=num_test_samples, num_cb_samples=num_cb_samples, MODE=MODE, LOAD_DATA_DIR=LOAD_DATA_DIR, dist=DIST)

            # Generate initial predictions
            emb_initialiser, initial_weights = setup_embedding_weights(X_test[1], DISTRACTOR, trainable_params, DIST=DIST, OFFSET_NT=OFFSET_NT, RESET_PRED_TO_ZERO=RESET_PRED_TO_ZERO, log_path=logs_dirs[0])

            # Create the visualisation callbacks
            epoch_pred_cbs, test_logging_cbs = initialise_pred_cbs(logs_dirs, test_vis_dirs, smpl, X_cb, silh_cb, trainable_params, LEARNING_RATES, ARCHITECTURE)
            epoch_pred_cbs_control, test_logging_cbs_control = initialise_pred_cbs(control_logs_dirs, control_dirs, smpl, X_cb, silh_cb, trainable_params, LEARNING_RATES, ARCHITECTURE)


            """ Test run """
            print(test_vis_model_dir)

            for lr_num, lr in enumerate(LEARNING_RATES):
                # Set-up the model
                optlearner_model = setup_test_model(model, ARCHITECTURE, data_samples, param_trainable, emb_initialiser, initial_weights, INPUT_TYPE, JOINT_LEVELS, update_weight, lr, points_lr_weights[lr_num], normals_lr_weights[lr_num], OPTIMIZER, INCLUDE_SHAPE)

                # Reset the weights of the embedding layer
                print("Resetting predictions...")
                optlearner_model = initialise_emb_layers(optlearner_model, param_trainable, initial_weights)

                # Perform full optimisation for this learning rate
                print("\nTesting for lr '{:03f}'...".format(lr))
                learned_optimizer_gradient_updates(optlearner_model, X_test, Y_test, param_trainable, epoch_pred_cbs[lr_num], test_logging_cbs[lr_num], num_samples=num_test_samples, epochs=TEST_EPOCHS, mode=MODE, lr=lr, BATCH_SIZE=TEST_BATCH_SIZE)

                test_type = get_test_type(lr, OPTIMIZER)

                # Reset session in preparation for next test distribution
                clear_session()
            #regular_optimizer(optlearner_model, epochs=TEST_EPOCHS)
            plot_metrics([test_vis_model_dir], setup=test_type, show=False)

