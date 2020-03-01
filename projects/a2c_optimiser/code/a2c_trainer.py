from __future__ import print_function

import os
import sys
import argparse
from datetime import datetime
import keras
from keras.models import Model
from keras.optimizers import Adam, SGD
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import yaml
import numpy as np

from environments.A2CEnv import A2CEnv
from architectures.DenseValueNetwork import DenseValueNetwork
from architectures.ResConv1DPolicyNetwork import ResConv1DPolicyNetwork


""" General set-up """

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

# Create experiment directory
exp_dir = "/data/cvfs/hjhb2/projects/a2c_optimiser/experiments/" + str(run_id) + "/"
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
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = setup_params["GENERAL"]["GPU_ID"]
print("gpu used:|" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")


""" Environment set-up and data extraction """

# Experimental configuration parameters
RESET_PERIOD = setup_params["BASIC"]["RESET_PERIOD"]
MODEL_SAVE_PERIOD = setup_params["BASIC"]["MODEL_SAVE_PERIOD"]
PREDICTION_PERIOD = setup_params["BASIC"]["PREDICTION_PERIOD"]
OPT_LR = setup_params["GD_OPT"]["OPT_LR"]
OPT_ITER = setup_params["GD_OPT"]["OPT_ITER"]
REWARD_FACTOR = setup_params["GD_OPT"]["REWARD_FACTOR"]
EPSILON = setup_params["GD_OPT"]["EPSILON"]
PARAM_OFFSET = setup_params["BASIC"]["PARAM_OFFSET"]
PARAMS_TO_OFFSET = setup_params["BASIC"]["PARAMS_TO_OFFSET"]
TARGET_OFFSET = setup_params["DATA"]["TARGET_OFFSET"]
TARGET_PARAMS_TO_OFFSET = setup_params["DATA"]["TARGET_PARAMS_TO_OFFSET"]
#ARCHITECTURE = setup_params["MODEL"]["ARCHITECTURE"]
BATCH_SIZE = setup_params["MODEL"]["BATCH_SIZE"]


# Instantiation of the task environment
env = A2CEnv(TARGET_OFFSET, TARGET_PARAMS_TO_OFFSET, PARAM_OFFSET, PARAMS_TO_OFFSET, SMPL, BATCH_SIZE=BATCH_SIZE, opt_lr=OPT_LR, opt_iter=OPT_ITER, reward_factor=REWARD_FACTOR, epsilon=EPSILON)



