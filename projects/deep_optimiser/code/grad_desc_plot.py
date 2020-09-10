
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
import csv

#from tools.data_helpers import format_offsetable_params
from tools.rotation_helpers import geodesic_error

# Parse the command-line arguments
parser = ArgumentParser()
parser.add_argument("--dirs", nargs="+", help="Path to first loss file")

args = parser.parse_args()

if args.dirs is not None:
    dirs = args.dirs
else:
    #dirs = ["../"]
    # Update trials
    # Gaussian (1.1/0.1), T-pose init. with gradient updates in training
    #dirs = ["/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-11_14:49:44/test_vis/model.1449-0.0054.hdf5_train_gaussian/"]
    # Gaussian (1.2/0.2), T-pose init. with gradient updates in training
    #dirs = ["/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-11_14:50:46/test_vis/model.1999-0.0207.hdf5_train_gaussian/"]
    #dirs += ["/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-11_14:50:46/test_vis/model.1999-0.0207.hdf5_train_gaussian/"]
    # Gaussian (1.2/0.2), T-pose init. with gradient updates in training, deeper network
    #dirs = ["/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-12_13:11:18/test_vis/model.499-0.0201.hdf5_train_gaussian/"]
    dirs = ["/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-13_14:14:16/test_vis/model.05-0.0936.hdf5_test_gaussian_zero_init/"]

    # Adam grad-desc trials
    #dirs = ["/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-12_20:57:57/test_vis/model.04-0.0862.hdf5_train_gaussian/"]
    # SGD grad-desc trials
    #dirs = ["/data/cvfs/hjhb2/projects/deep_optimiser/experiments/GroupedConv1DOptLearnerArchitecture_2020-05-12_20:57:57/test_vis/model.05-0.0839.hdf5_train_gaussian/"]

#exp_method = ["points", "norm", "update"]
exp_method = ["update"]
#exp_method = ["points"]
#exp_method = ["norm"]

lrs_to_plot = [1.0, 0.5, 0.125]

# Presets
method_styles = {"points": ".", "norm": "+", "update": "v"}
#lr_styles = {4.0: "c", 2.0: "m", 1.0: "r", 0.5: "g", 0.125: "b"}
color_styles = ["c", "m"]
metrics_styles = {"mse": '-', "median": "-.", "ang_metric": '--', "ang_median": "dotted"}

def get_values_from_file(directory):
    files = os.listdir(directory)
    #print(files)
    files = sorted([f for f in files if "data_E" in f], key=lambda x: int(x.replace("data_E", "").replace(".npz", "")))
    #print(files)
    history = []
    for filename in files:
        with open(directory + "/" + filename) as f:
            values = dict(np.load(f))
            history.append(values)
        #print(values)
    return history


def get_metrics_from_file(directory):
    history = get_values_from_file(directory)

    mse_history = []
    median_history = []
    ang_metric_history = []
    ang_median_history = []
    for entry in history:
        median = np.median(np.square(entry["delta_d"]))
        mse = np.mean(np.square(entry["delta_d"]))
        mse_history.append(mse)
        median_history.append(median)

        angular_errors = geodesic_error(entry["gt_params"], entry["opt_params"])
        angular_metric = np.mean(angular_errors)
        angular_metric_median = np.median(angular_errors)
        ang_metric_history.append(angular_metric)
        ang_median_history.append(angular_metric_median)

    return mse_history, median_history, ang_metric_history, ang_median_history


grad_desc_runs = []
for i, log_dir in enumerate(dirs):
    name_string = "D" + str(i)
    dir_exp = os.listdir(log_dir)
    dir_exp_dict = {method: {} for method in exp_method}

    print("Collecting data...")
    for exp in dir_exp:
        lower_exp = exp.lower()
        #print(lower_exp)

        lower_exp_in_methods = [method in lower_exp for method in exp_method]
        if np.sum(lower_exp_in_methods) == 1:
            method = np.array(exp_method)[lower_exp_in_methods][0]
            lr = float(lower_exp[lower_exp.find("lr")+2:])

            if lr in lrs_to_plot:
                # Get values of data during training
                mse_history, median_history, ang_metric_history, ang_median_history = get_metrics_from_file(log_dir + exp)

                # Add to data dictionary
                method_lr_values = {lr: {"mse": mse_history, "median": median_history, "ang_metric": ang_metric_history, "ang_median": ang_median_history}}
                #method_lr_values = {lr: {"mse": mse_history, "ang_metric": ang_metric_history}}
                dir_exp_dict[method].update(method_lr_values)

    print("Plotting data...")
    for method, lr_exp_metrics in dir_exp_dict.items():
        for lr, metrics in lr_exp_metrics.items():
            for metric_type, metric_values in metrics.items():
                plot_label = "{}, {}, LR: {}, {}".format(name_string, method, lr, metric_type)
                #plt.plot(range(len(metric_values)), metric_values, markersize=2, linewidth=1, marker=method_styles[method], color=lr_styles[lr], linestyle=metrics_styles[metric_type], label=plot_label)
                plt.plot(range(len(metric_values)), metric_values, markersize=2, linewidth=1, marker=method_styles[method], color=color_styles[i], linestyle=metrics_styles[metric_type], label=plot_label)


plt.ylabel("Metric value", fontsize=16)
plt.xlabel("Iteration", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(prop={'size': 10})
#plt.savefig("/data/cvfs/hjhb2/projects/deep_optimiser/plots/plot_" + str(datetime.datetime.now()) + ".png")
plt.show()



