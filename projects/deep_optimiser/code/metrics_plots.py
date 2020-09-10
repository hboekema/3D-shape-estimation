
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
import pickle

#from tools.data_helpers import format_offsetable_params
from tools.rotation_helpers import geodesic_error


def store_best_results(curr, other, lr, eval_on="ang_median"):
    iterations_to_store = curr.keys().remove("lr")
    print(iterations_to_store)
    curr_best = [curr[iter_][eval_on] for iter_ in sorted(iterations_to_store)]
    other_best = other[eval_on][iterations_to_store]

    if np.min(other_best) < np.min(curr_best):
        for iter_, iter_metrics in curr.items():
            if iter_ != "lr":
                for metric, metric_value in iter_metrics.items():
                    curr[iter_][metric] = other[metric][iter_]
        curr["lr"] = lr

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
    percentile_history = []
    ang_metric_history = []
    ang_median_history = []
    ang_percentile_history = []
    for entry in history:
        delta_d = entry["gt_params"] - entry["opt_params"]
        median = np.median(np.square(delta_d))
        mse = np.mean(np.square(delta_d))
        percentile = np.percentile(np.square(delta_d), q=10, interpolation="higher")
        mse_history.append(mse)
        median_history.append(median)
        percentile_history.append(percentile)

        angular_errors = geodesic_error(entry["gt_params"], entry["opt_params"])
        joints_to_exclude = []
        #joints_to_exclude = [7, 8, 10, 11, 20, 21, 22, 23]
        #joints_to_exclude = [7, 8, 10, 11, 12, 15, 20, 21, 22, 23]
        angular_errors_filtered = angular_errors[:, [i for i in range(24) if i not in joints_to_exclude]]
        #print("selected joints: " + str(np.mean(angular_errors_filtered)))
        #exit(1)
        angular_errors = angular_errors_filtered
        angular_metric = np.mean(angular_errors)
        #print("all joints: " + str(angular_metric) + "\n")
        angular_metric_median = np.median(angular_errors)
        ang_percentile = np.percentile(angular_errors, q=10, interpolation="higher")
        #ang_perc_per_joint = np.percentile(angular_errors, q=10, interpolation="higher", axis=0)
        #print(ang_perc_per_joint)
        #exit(1)
        ang_metric_history.append(angular_metric)
        ang_median_history.append(angular_metric_median)
        ang_percentile_history.append(ang_percentile)

    return mse_history, median_history, percentile_history, ang_metric_history, ang_median_history, ang_percentile_history


def get_per_param_metrics_from_file(directory):
    history = get_values_from_file(directory)

    angular_metric_history = []
    for entry in history:
        angular_errors = geodesic_error(entry["gt_params"], entry["opt_params"])
        angular_metric = np.mean(angular_errors, axis=0)
        angular_metric_history.append(angular_metric)

    return angular_metric_history


def get_percentiles_from_file(directory, percentiles):
    history = get_values_from_file(directory)

    angular_metrics = []
    angular_errors_history = []
    for entry in history:
        angular_errors = geodesic_error(entry["gt_params"], entry["opt_params"])
        metric = np.mean(angular_errors)
        angular_metrics.append(metric)
        angular_errors_history.append(np.mean(angular_errors, axis=1))


    index = angular_metrics.index(np.min(angular_metrics))
    print(index)
    best_iter_errors = np.array(angular_errors_history[index])

    percentile_bins = []
    for percentile in percentiles:
        num_in_percentile = np.sum(best_iter_errors < percentile)
        percentile_bins.append(num_in_percentile)

    percentile_bins = np.array(percentile_bins, dtype=float) / float(best_iter_errors.shape[0])
    print(percentile_bins)

    np.save("sgd_ROC.npy", percentile_bins)
    #np.save("adam_ROC.npy", percentile_bins)
    #np.save("deep_opt_ROC.npy", percentile_bins)

    plt.plot(percentiles, percentile_bins)
    plt.xlabel("Angular error (radians)", fontsize=16, fontweight="bold")
    plt.ylabel("Percentage of samples", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.show()



def split_losses(losses, kinematic_levels=None):
    # Group losses, if groups are specified
    #level_names = ["level_{}".format(i) for i in range(len(kinematic_levels))]
    level_names = ["feet", "hands", "neck", "back", "other"]

    levelled_losses = []
    for level in kinematic_levels:
        level_mean = np.mean(np.array(losses)[:, level], axis=1)
        #print(level_mean.shape)
        levelled_losses.append(level_mean)

    #print(levelled_losses)

    # Choose colours for plot
    num_plots = len(levelled_losses)
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))

    # Plot the losses for each parameter
    for i, losses in enumerate(levelled_losses):
        plt.plot(range(len(losses)), losses, label=level_names[i])

    plt.xlabel('Iteration', fontsize=16, fontweight='bold')
    plt.ylabel('Angular error', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.legend(prop={'size': 14, 'weight': 'bold'})
    plt.show()

    return levelled_losses



def results_dict_to_file(results_dict, output_dir, iterations_to_store):
    test_type = results_dict.keys()[0]

    results = {}
    for lr, lr_metrics in results_dict[test_type].items():
        for metric, metric_values in lr_metrics.items():
            for i, value in enumerate(metric_values):
                key = (iterations_to_store[i], metric)
                if key not in results.keys():
                    results[key] = {}
                results[key].update({float(lr.replace("lr_", "")): value})

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(output_dir + "{}_performance.csv".format(test_type))



def plot_metrics(dirs, setup, show=True, input_type="3D"):
    if setup == "adam_gd":
        #exp_method = ["points", "norm"]
        exp_method = ["points"]
        #exp_method = ["norm"]

        #lrs_to_plot = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        #lrs_to_plot = [0.5, 0.05, 0.005]
        #lrs_to_plot = [0.1, 0.05, 0.01]
        lrs_to_plot = [0.05]
        lr_styles = {0.0005: "c", 0.005: "m", 0.0001: "r", 0.5: "g", 0.1: "y", 0.125: "b", 0.01: "orange", 0.001: "k", 0.05: "limegreen", 0.1: "navy", 0.5: "indigo", 1.0: "slategrey"}
        optimizers = ["Adam"]

    elif setup == "sgd_gd":
        #exp_method = ["points", "norm"]
        exp_method = ["points"]
        #exp_method = ["norm"]

        #lrs_to_plot = [4.0, 2.0, 1.5, 1.0, 0.5, 0.1, 0.125, 0.05, 0.01, 0.005, 0.001]
        #lrs_to_plot = [4.0, 2.0, 1.5, 1.0, 0.5, 0.125, 0.1, 0.05, 0.01]
        #lrs_to_plot = [2.0, 1.5, 1.0, 0.5]
        #lrs_to_plot = [1.5, 1.0, 0.5]
        lrs_to_plot = [1.0]
        lr_styles = {4.0: "slategrey", 2.0: "limegreen", 1.5: "indigo",  0.05: "m", 1.0: "r", 0.5: "g", 0.1: "y", 0.125: "b", 0.01: "orange", 0.001: "k", 0.005: "navy"}
        optimizers = ["SGD"]

    else:
        exp_method = ["update"]
        #lrs_to_plot = [1.0, 0.5, 0.125]
        lrs_to_plot = [0.125]
        lr_styles = {0.0005: "c", 0.005: "m", 1.0: "r", 0.5: "g", 0.1: "y", 0.125: "b", 0.01: "orange", 0.001: "k"}
        optimizers = ["SGD", "Adam"]


    #metrics_to_show = ["mean", "median", "percentile"]
    #metrics_to_show = ["mean"]
    metrics_to_show = ["median", "mean"]
    #metrics_to_show = ["median", "percentile"]

    # Presets
    method_styles = {"Adam": {"points": ".", "norm": "+", "update": "v"}, "SGD": {"points": "<", "norm": ">", "update": "s"}}
    color_styles = ["c", "m"]
    metrics_styles = {"mse": '-', "ang_metric": 'dotted', "median": "dashdot", "ang_median": "dashed", "percentile": (0, (5, 10)), "ang_percentile": (0, (1, 10))}
    markersize = {"Adam": 6, "SGD": 4}

    iterations_to_show = 200
    iterations_to_store = [10, 100]

    grad_desc_runs = []
    stored_iterations = {setup: {}}
    for i, test_dir in enumerate(dirs):
        name_string = "D" + str(i)
        dir_exp = os.listdir(test_dir)
        dir_exp_dict = {opt: {} for opt in optimizers}

        print("Collecting data...")
        for exp in dir_exp:
            lower_exp = exp.lower()
            #print(lower_exp)

            opt_mask = [opt.lower() in lower_exp for opt in optimizers]
            if np.sum(opt_mask) == 1 and "lr" in lower_exp:
                exp_opt = np.array(optimizers)[opt_mask][0]

                lower_exp_in_methods = [method in lower_exp for method in exp_method]
                if np.sum(lower_exp_in_methods) == 1:
                    method = np.array(exp_method)[lower_exp_in_methods][0]
                    lr = float(lower_exp[lower_exp.find("lr")+2:])
                    print(lr)
                    if lr in lrs_to_plot:
                        # Get values of data during training
                        #percentiles = np.linspace(0, 1, num=100)
                        #percentiles_binned = get_percentiles_from_file(test_dir + exp, percentiles)
                        #exit(1)

                        mse_history, median_history, percentile_history, ang_metric_history, ang_median_history, ang_percentile_history = get_metrics_from_file(test_dir + exp)
                        losses_history = get_per_param_metrics_from_file(test_dir + exp)


                        #kinematic_levels = [
                        #        [7,8,10,11],
                        #        [20,21,22,23],
                        #        [12, 15],
                        #        [6, 9],
                        #        [0,1,2,3,4,5,13,14,16,17,18,19]
                        #        ]
                        #split_loss_history = split_losses(losses_history, kinematic_levels)

                        if len(mse_history) > 0:

                            new_results = {"mse":mse_history, "median":median_history, "ang_metric":ang_metric_history, "ang_median":ang_median_history}
                            #new_results = {"ang_metric":ang_metric_history, "ang_median":ang_median_history, "ang_perc":ang_percentile_history}
                            new_results_values = {key: list(np.array(value)[iterations_to_store]) + [np.min(value), value.index(np.min(value))] for key, value in new_results.items()}
                            stored_iterations[setup].update({"lr_{}".format(lr): new_results_values})

                            # Add to data dictionary
                            metrics_to_store = {}
                            if "mean" in metrics_to_show:
                               #metrics_to_store.update({"mse": mse_history, "ang_metric": ang_metric_history})
                               metrics_to_store.update({"ang_metric": ang_metric_history})
                            if "median" in metrics_to_show:
                               #metrics_to_store.update({"median": median_history, "ang_median": ang_median_history})
                               metrics_to_store.update({"ang_median": ang_median_history})
                            if "percentile" in metrics_to_show:
                               metrics_to_store.update({"percentile": percentile_history, "ang_percentile": ang_percentile_history})


                            method_lr_values = {(method, lr): metrics_to_store}
                            dir_exp_dict[exp_opt].update(method_lr_values)

        print("Results dict: " + str(stored_iterations))
        results_dict_to_file(stored_iterations, test_dir, iterations_to_store + ["min", "min_epoch"])

        print("Plotting data...")
        fig, ax = plt.subplots()
        for opt, opt_metrics in sorted(dir_exp_dict.items()):
                for metrics_lr_tuple, metrics in sorted(opt_metrics.items()):
                    method = metrics_lr_tuple[0]
                    lr = metrics_lr_tuple[1]
                    for metric_type, metric_values in sorted(metrics.items()):
                        #plot_label = "{}, {}, {}, LR: {}, {}".format(name_string, opt, method, lr, metric_type)
                        prettified_metric = metric_type
                        if metric_type == "ang_metric":
                            prettified_metric = "Ang. Err. (MSE)"
                        elif metric_type == "ang_median":
                            prettified_metric = "Ang. Err. (Med.)"

                        if method == "update":
                            plot_label = "Deep Opt. - {} - LR: {} - {}".format(input_type, lr, prettified_metric)
                        else:
                            plot_label = "{} - {} - LR: {} - {}".format(opt, input_type, lr, prettified_metric)

                        if len(metric_values) > iterations_to_show:
                            metric_values = metric_values[:iterations_to_show]

                        ax.plot(range(len(metric_values)), metric_values, markersize=markersize[opt], markevery=5, linewidth=1, marker=method_styles[opt][method], color=lr_styles[lr], linestyle=metrics_styles[metric_type], label=plot_label)


    plt.ylabel("Metric value", fontsize=16, fontweight='bold')
    plt.xlabel("Iteration", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.legend(prop={'size': 14, 'weight': 'bold'})
    for dir_ in dirs:
        fig.savefig(dir_ + str(datetime.datetime.now()) + ".png")
        pickle.dump(fig, open(dir_ + str(datetime.datetime.now()) + ".pickle", "wb"))

    if show:
        plt.show()


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--dirs", nargs="+", help="Path to first loss file")

    args = parser.parse_args()

    if args.dirs is not None:
        dirs = args.dirs
    else:
        dirs = ["../"]

    setup = "deep_opt"
    #setup = "adam_gd"
    #setup = "sgd_gd"

    #input_type = "2D"
    input_type = "3D"

    plot_metrics(dirs, setup, show=True, input_type=input_type)



