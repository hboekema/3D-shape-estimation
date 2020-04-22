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

from smpl_np import SMPLModel
from environments.A2CEnv import A2CEnv
from A2C.A2C import A2C
#from A2C.agent import agent
#from A2C.actor import actor
#from A2C.critic import critic
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
# Basic config
MODEL_SAVE_PERIOD = setup_params["BASIC"]["MODEL_SAVE_PERIOD"]
PREDICTION_PERIOD = setup_params["BASIC"]["PREDICTION_PERIOD"]
# Data config
PARAM_OFFSET = setup_params["DATA"]["PARAM_OFFSET"]
PARAMS_TO_OFFSET = setup_params["DATA"]["PARAMS_TO_OFFSET"]
TARGET_OFFSET = setup_params["DATA"]["TARGET_OFFSET"]
TARGET_PARAMS_TO_OFFSET = setup_params["DATA"]["TARGET_PARAMS_TO_OFFSET"]
# General model config
BATCH_SIZE = setup_params["MODEL"]["BATCH_SIZE"]
GAMMA = setup_params["MODEL"]["GAMMA"]
NUM_EPISODES = setup_params["MODEL"]["NUM_EPISODES"]
K = setup_params["MODEL"]["K"]
# Environment config
OPT_LR = setup_params["ENV"]["OPT_LR"]
OPT_ITER = setup_params["ENV"]["OPT_ITER"]
REWARD_SCALE = setup_params["ENV"]["REWARD_SCALE"]
REWARD_FACTOR = setup_params["ENV"]["REWARD_FACTOR"]
STEP_LIMIT = setup_params["ENV"]["STEP_LIMIT"]
EPSILON = setup_params["ENV"]["EPSILON"]
# Actor config
ACTOR_LR = setup_params["ACTOR"]["ACTOR_LR"]
# Critic config
CRITIC_LR = setup_params["CRITIC"]["CRITIC_LR"]

# Set random seed for reproducibility
np.random.seed(10)

# Process data modifiers
def param_mapping(PARAMS_TO_OFFSET):
    """ Map special named combinations """
    param_ids = ["param_{:02d}".format(i) for i in range(85)]
    not_trainable = []
    if PARAMS_TO_OFFSET == "all_pose":
        not_trainable = [0, 1, 2]
    elif PARAMS_TO_OFFSET == "all_pose_and_global_rotation":
        not_trainable = [0, 2]

    trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 66]
    PARAMS_TO_OFFSET = [param_ids[index] for index in trainable_params_indices]
    return PARAMS_TO_OFFSET

PARAMS_TO_OFFSET = param_mapping(PARAMS_TO_OFFSET)
TARGET_PARAMS_TO_OFFSET = param_mapping(TARGET_PARAMS_TO_OFFSET)

# Instantiate the SMPL model
SMPL = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

# Instantiation of the task environment
env = A2CEnv(TARGET_OFFSET, TARGET_PARAMS_TO_OFFSET, PARAM_OFFSET, PARAMS_TO_OFFSET, SMPL, BATCH_SIZE=BATCH_SIZE, opt_lr=OPT_LR, opt_iter=OPT_ITER, reward_factor=REWARD_FACTOR, reward_scale=REWARD_SCALE, step_limit=STEP_LIMIT, epsilon=EPSILON)

# Instantiation of the actor and critic network base
env_dim = (6890, 3)
actor_input, actor_output = ResConv1DPolicyNetwork(env_dim)
critic_input, critic_output = DenseValueNetwork(env_dim)
actor_base = Model(actor_input, actor_output)
critic_base = Model(critic_input, critic_output)

# Instantiation of the A2C algorithm
act_dim = (85,)
a2c_model = A2C(act_dim, env_dim, actor_base, critic_base, k=K, batch_size=BATCH_SIZE, gamma=GAMMA, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)


if __name__ == "__main__":
    results = a2c_model.train(env, NUM_EPISODES, save_period=MODEL_SAVE_PERIOD, save_dir=model_dir)

    print(results)

