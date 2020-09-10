import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from copy import copy
import datetime
import yaml

from tools.data_helpers import format_offsetable_params


# Parse the command-line arguments
parser = ArgumentParser()
parser.add_argument("--dirs", nargs="+", help="Path to first loss file")

args = parser.parse_args()

if args.dirs is not None:
    dirs = args.dirs
else:
    dirs = ["../"]

dir_paths = dirs
dirs = [dir_name + "logs/params_mspe.txt" for dir_name in dirs]

dirs_trainable_params = []
for dir_name in dir_paths:
    with open(dir_name + "code/config.yaml", 'r') as f:
        try:
            setup_params = yaml.safe_load(f)
            #print(setup_params)
        except yaml.YAMLError as exc:
            print(exc)
    trainable_params_ = setup_params["PARAMS"]["TRAINABLE"]
    trainable_params_ = format_offsetable_params(trainable_params_)
    dirs_trainable_params.append(trainable_params_)

trainable_params = dirs_trainable_params[0]

#kinematic_levels = None
kinematic_levels = [
            [0],
            [3],
            [1,2,6],
            [4,5,9],
            [7,8,12,13,14],
            [15,16,17],
            [18,19],
            [20,21],
            [22,23]
            ]


SUBSAMPLE_PERIOD = 20
#SUBSAMPLE_PERIOD = 10
#SUBSAMPLE_PERIOD = 2

def add_losses(dir_losses, epoch_array, curr_epoch):
    epoch_str = "".join(epoch_array)
    #print(epoch_str)

    epoch_losses = epoch_str.split(" ")
    epoch_losses = [i for i in epoch_losses if i != ""]
    #print(epoch_losses)
    epoch_losses = [float(num) for num in epoch_losses]
    #print(epoch_losses)

    dir_losses[curr_epoch] = epoch_losses
    epoch_array = []

    return dir_losses, epoch_array


# Load the loss files
for i, dir_name in enumerate(dirs):
    dir_losses = {}

    with open(dir_name, 'r') as infile:
        epoch_array = []
        curr_epoch = None
        for line in infile:
            if "epoch" in line.strip():
                epoch_num = int(line.replace("epoch ", ""))
                #print(epoch_num)

                if curr_epoch is not None:
                    dir_losses, epoch_array = add_losses(dir_losses, epoch_array, curr_epoch)
                curr_epoch = epoch_num
            else:
                line = line.replace("\n", "")
                line = line.replace("[", "").replace("]", "")
                epoch_array.append(line)

        if curr_epoch is not None:
            dir_losses, epoch_array = add_losses(dir_losses, epoch_array, curr_epoch)


# Group losses, if groups are specified
level_names = None
if kinematic_levels is None:
    level_names = ["param_{:02d}".format(i) for i in range(72)]
    level_names = [name for name in level_names if name in trainable_params]
    kinematic_levels = [[i] for i in range(24)]
else:
    level_names = ["level_{}".format(i) for i in range(len(kinematic_levels))]
    to_pop = []
    for i, level in enumerate(kinematic_levels):
        level_params = []
        for joint in level:
            j1 = 3*joint
            j2 = j1 + 1
            j3 = j2 + 1

            curr_joints = [j1, j2, j3]
            curr_joints = [j for j in curr_joints if "param_{:02d}".format(j) in trainable_params]

            level_params += curr_joints
        level_params = ["param_{:02d}".format(param) for param in level_params]
        #print(level_params)
        intersection = set(level_params).intersection(set(trainable_params))
        if len(intersection) == 0:
            to_pop.append(i)

    for index in sorted(to_pop, reverse=True):
        level_names.pop(index)
    print(level_names)

# Gather losses for each group
levelled_indices = []
for i, level in enumerate(kinematic_levels):
    level_indices = []
    for joint in level:
        j1 = 3*joint
        j2 = j1 + 1
        j3 = j2 + 1

        curr_joints = [j1, j2, j3]
        curr_joints = [j for j in curr_joints if "param_{:02d}".format(j) in trainable_params]

        level_indices += curr_joints

    levelled_indices.append(level_indices)
print(levelled_indices)

levelled_losses = {}
epochs = []
for epoch, losses in dir_losses.items():
    losses = np.array(losses)
    epoch_losses = []
    for indices in levelled_indices:
        if len(indices) > 0:
            epoch_losses.append(np.mean(losses[indices]))
    levelled_losses[epoch] = epoch_losses
    epochs.append(epoch)

epochs = sorted(epochs)
#print(levelled_losses)

# Plot these losses
loss_values = np.array([levelled_losses[key] for key in sorted(levelled_losses.keys())])
print("loss_values shape: " + str(loss_values.shape))
loss_values = loss_values.T
print("loss_values shape: " + str(loss_values.shape))

# Choose colours for plot
num_plots = loss_values.shape[0]
colormap = plt.cm.gist_ncar
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))

# Plot the losses for each parameter
for i, group_losses in enumerate(loss_values):
    plt.plot(epochs, group_losses, label=level_names[i])

plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.show()

