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
    config_path = args.config
else:
    config_path = "./config.yaml"

with open(config_path, 'r') as f:
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
print("Keras version: " + str(keras.__version__))
print("TF version: " + str(K.tf.__version__))
import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

from tools.training_helpers_v3 import train_model


if __name__ == "__main__":
    np.random.seed(10)

    optlearner_model = train_model(setup_params, run_id)

