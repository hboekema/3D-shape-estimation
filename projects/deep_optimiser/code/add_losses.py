
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

from tools.data_helpers_v2 import format_offsetable_params

# Parse the command-line arguments
parser = ArgumentParser()
parser.add_argument("--dirs", nargs="+", help="Path to first loss file")
parser.add_argument("-e", action="store_true", help="Extra (2D) losses to be plotted")

args = parser.parse_args()

if args.dirs is not None:
    dirs = args.dirs
else:
    dirs = ["../"]

dir_paths = dirs
dirs_trainable_joints = []
for dir_path in dir_paths:
    with open(dir_path + "code/config.yaml", 'r') as f:
        try:
            setup_params = yaml.safe_load(f)
            #print(setup_params)
        except yaml.YAMLError as exc:
            print(exc)
    trainable_params = setup_params["PARAMS"]["TRAINABLE"]
    trainable_params = format_offsetable_params(trainable_params)
    trainable_params = [int(param.replace("param_", "")) for param in trainable_params]
    trainable_joints = sorted(np.unique([int((param - (param % 3))/3) for param in trainable_params]))
    #print(trainable_joints)
    #exit(1)
    dirs_trainable_joints.append(trainable_joints)

dirs = [dir_name + "logs/" for dir_name in dirs]

if args.e:
    losses_to_add = ["params_angle", "delta_angle", "gt_normals_LOSS"]
else:
    losses_to_add = ["params_angle", "delta_angle"]
losses_to_filter = ["params_angle"]
losses_to_add_and_filter = ["delta_angle"]

def add_losses(dir_losses, epoch_array, curr_epoch, filter_=None):
    epoch_str = "".join(epoch_array)
    #print(epoch_str)

    epoch_losses = epoch_str.split(" ")
    epoch_losses = [i for i in epoch_losses if i != ""]
    #print(epoch_losses)
    epoch_losses = np.array([float(num) for num in epoch_losses])
    #print(epoch_losses)
    #print(epoch_losses.shape)
    if filter_ is not None:
        epoch_losses = epoch_losses[filter_]
    #print(epoch_losses)

    dir_losses[curr_epoch] = np.mean(epoch_losses)

    return dir_losses


for i, dir_ in enumerate(dirs):
    loss_values_to_add = {}
    for loss in losses_to_add:
        if loss in losses_to_filter or loss in losses_to_add_and_filter:
            filter_ = dirs_trainable_joints[i]
        else:
            filter_ = None

        dir_losses = {}
        dir_losses_2 = {}

        with open(dir_ + loss + ".txt", 'r') as infile:
            epoch_array = []
            curr_epoch = None
            for line in infile:
                if "epoch" in line.strip():
                    epoch_num = int(line.replace("epoch ", ""))
                    #print(epoch_num)

                    if curr_epoch is not None:
                        dir_losses = add_losses(dir_losses, epoch_array, curr_epoch, filter_)
                        if loss in losses_to_add_and_filter:
                            dir_losses_2 = add_losses(dir_losses_2, epoch_array, curr_epoch, None)
                        epoch_array = []

                    curr_epoch = epoch_num
                else:
                    line = line.replace("\n", "")
                    line = line.replace("[", "").replace("]", "")
                    epoch_array.append(line)

            if curr_epoch is not None:
                dir_losses = add_losses(dir_losses, epoch_array, curr_epoch, filter_)
                if loss in losses_to_add_and_filter:
                    dir_losses_2 = add_losses(dir_losses_2, epoch_array, curr_epoch, None)
            epoch_array = []

        if loss in losses_to_add_and_filter:
            loss_values_to_add[loss] = dir_losses_2
            loss_values_to_add[loss + "_filtered"] = dir_losses
        else:
            loss_values_to_add[loss] = (dir_losses)


    # Now amend the full loss file
    amended_losses = open(dir_ + "losses_ext.txt", "w")
    losses_file = open(dir_ + "losses.txt", "r")

    prev_values = {}
    for i, s in enumerate(losses_file.readlines()):
        if i > 0:
            s = json.loads(s)
            epoch = s["epoch"]
            for loss, value in loss_values_to_add.items():
                if epoch in value.keys():
                    s[loss] = value[epoch]
                    prev_values[loss] = value[epoch]
                elif loss in prev_values.keys():
                    s[loss] = prev_values[loss]
                else:
                    s[loss] = None

            json.dump(s, amended_losses)
            amended_losses.write('\n')

    amended_losses.close()
    losses_file.close()


