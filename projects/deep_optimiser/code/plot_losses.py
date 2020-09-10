import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from copy import copy
import datetime


# Parse the command-line arguments
parser = ArgumentParser()
parser.add_argument("--dirs", nargs="+", help="Path to first loss file")
parser.add_argument("-e", action="store_true", help="Extra (2D) losses to be plotted")

args = parser.parse_args()

if args.dirs is not None:
    dirs = args.dirs
else:
    dirs = ["../"]

#print(args.e)

if not args.e:
    style_presets = [".", "+", "v", "s", ","]
    color_presets = ["r", "b", "g", "k", "y", "m", "m", "c", "limegreen"]
    linestyles = ["-", "--", "-", "-", "-", "-", "--", "--", "-"]

    #dirs = [dir_name + "logs/losses.txt" for dir_name in dirs]
    dirs = [dir_name + "logs/losses_ext.txt" for dir_name in dirs]

    losses_to_load = ["loss", "delta_d_hat_mse_loss", "pc_mean_euc_dist_loss", "delta_d_mse_loss", "diff_angle_mse_loss", "delta_angle", "delta_angle_filtered", "params_angle", "update_loss_loss"]
    loss_alias = ["loss", "deep opt. loss", "point cloud loss", "smpl loss", "angle loss", "smpl angle", "smpl_angle_filtered", "preds. angle", "update loss"]
    loss_display_name = [alias + " (" + losses_to_load[i] + ")" for i, alias in enumerate(loss_alias)]
    scale_presets = [0.001, 1, 1, 1, 1, 1, 1, 1, 1]

else:
    style_presets = [".", "+", "v", "s", ","]
    color_presets = ["r", "b", "g", "k", "y", "m", "m", "c", "orange", "limegreen"]
    linestyles = ["-", "--", "-", "-", "-", "-", "--", "--", "-", "-"]

    dirs = [dir_name + "logs/losses_ext.txt" for dir_name in dirs]

    #losses_to_load = ["loss", "delta_d_hat_mse_loss", "pc_mean_euc_dist_loss", "delta_d_mse_loss", "diff_angle_mse_loss"]
    #losses_to_load = ["loss", "delta_d_hat_mse_loss", "pc_mean_euc_dist_loss", "delta_d_mse_loss", "diff_angle_mse_loss", "delta_angle", "params_angle"]
    losses_to_load = ["loss", "delta_d_hat_mse_loss", "pc_mean_euc_dist_loss", "delta_d_mse_loss", "diff_angle_mse_loss", "delta_angle", "delta_angle_filtered", "params_angle", "gt_normals_LOSS", "update_loss_loss"]
    loss_alias = ["loss", "deep opt. loss", "point cloud loss", "smpl loss", "angle loss", "smpl angle", "smpl_angle_filtered", "preds. angle", "GT normals pred. loss", "update loss"]
    loss_display_name = [alias + " (" + losses_to_load[i] + ")" for i, alias in enumerate(loss_alias)]
    scale_presets = [0.001, 1, 1, 1, 1, 1, 1, 1, 1, 1]

loss_array = []
column_names = []
styles = []
colors = []
#SUBSAMPLE_PERIOD = 20
#SUBSAMPLE_PERIOD = 10
SUBSAMPLE_PERIOD = 5
#SUBSAMPLE_PERIOD = 2

# Load the loss files
for dir_num, dir_name in enumerate(dirs):
    dir_losses = []
    with open(dir_name, mode='r') as f:
        for i, s in enumerate(f.readlines()):
            if i > 0:
                s = json.loads(s)
                if s["epoch"] % SUBSAMPLE_PERIOD == 0:
                    dir_losses.append([s.get(loss) for loss in losses_to_load])
        loss_array.append(dir_losses)
        print("len of loss_array: " + str(len(loss_array)))
    column_names.append(["D" + str(dir_num + 1) + " " + name for name in loss_display_name])
    print(column_names)
    styles.append([style_presets[dir_num] for j in range(len(losses_to_load))])
    colors.append([color_presets[j] for j in range(len(losses_to_load))])

# Ensure that the length of all the arrays are the same
length_limit = np.min([len(arr) for arr in loss_array])
loss_array = [loss_arr[:length_limit] for loss_arr in loss_array]

column_names = np.concatenate(column_names)
styles = np.concatenate(styles)
colors = np.concatenate(colors)

df_list = []
for i, array in enumerate(loss_array):
    m = len(losses_to_load)
    print(len(column_names[(i*m):((i+1)*m)]))
    print(column_names)
    array = np.array(array)

    print(array.shape)
    df_list.append(pd.DataFrame(array, columns=column_names[(i*m):((i+1)*m)]))


#print(styles)
#print(colors)
#loss_array = np.concatenate(loss_array, axis=1)
#print(loss_array)
#loss_df = pd.DataFrame(loss_array, columns=column_names)
#print(loss_df)

for i, loss_df in enumerate(df_list):
    for j, column in enumerate(loss_df):
        print(loss_df[column])
        #plt.scatter(np.arange(len(loss_df))*SUBSAMPLE_PERIOD, loss_df[column], s=6, linewidths=1, marker=style_presets[i], color=color_presets[j])
        plt.plot(np.arange(len(loss_df))*SUBSAMPLE_PERIOD, scale_presets[j]*loss_df[column], markersize=6, linewidth=1, marker=style_presets[i], color=color_presets[j], linestyle=linestyles[j])
plt.ylabel("Loss value", fontsize=16)
plt.xlabel("Epoch", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(prop={'size': 10})
plt.savefig("/data/cvfs/hjhb2/projects/deep_optimiser/plots/plot_" + str(datetime.datetime.now()) + ".png")
plt.show()

