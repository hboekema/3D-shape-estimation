
import os
import numpy as np

from callbacks import OptLearnerPredOnEpochEnd


""" Directory operations """

def get_exp_models(exp_dir):
    print("Experiment directory: " + str(exp_dir))
    models = os.listdir(exp_dir + "models/")
    if len(models) == 0:
        print("No models for this experiment. Exiting.")
        exit(1)
    else:
        models.sort(key=lambda x: float(x[x.find("-")+1 : x.find(".hdf5")]))
        model_name = models[0]
        print("Using model '{}'".format(model_name))

    return model_name


def create_base_subdir(exp_dir, model_name, save_suffix=""):
    model = exp_dir + "models/" + model_name
    logs_dir = exp_dir + "logs/" + model_name + save_suffix + "/"
    os.system('mkdir ' + logs_dir)
    control_logs_dir = exp_dir + "logs/control" + save_suffix + "/"
    os.system('mkdir ' + control_logs_dir)
    test_vis_dir = exp_dir + "test_vis/" + model_name + save_suffix + "/"
    os.system('mkdir ' + test_vis_dir)
    control_dir = exp_dir + "test_vis/" + "control" + save_suffix + "/"
    os.system('mkdir ' + control_dir)

    return model, logs_dir, control_logs_dir, test_vis_dir, control_dir


def create_subdir(exp_dir, model_name, sub_dir, save_suffix=""):
    logs_dir = exp_dir + "logs/" + model_name + save_suffix + "/" + sub_dir
    os.system('mkdir ' + logs_dir)
    control_logs_dir = exp_dir + "logs/control" + save_suffix + "/" + sub_dir
    os.system('mkdir ' + control_logs_dir)
    test_vis_dir = exp_dir + "test_vis/" + model_name + save_suffix + "/" + sub_dir
    os.system('mkdir ' + test_vis_dir)
    control_dir = exp_dir + "test_vis/" + "control" + save_suffix + "/" + sub_dir
    os.system('mkdir ' + control_dir)

    return logs_dir, control_logs_dir, test_vis_dir, control_dir


def initialise_pred_cbs(logs_dirs, test_vis_dirs, smpl, X_cb, silh_cb, trainable_params, learning_rates):
    assert len(logs_dirs) == len(learning_rates)
    assert len(test_vis_dirs) == len(learning_rates)

    epoch_pred_cbs = []
    for lr_num, lr in enumerate(learning_rates):
        logs_dir = logs_dirs[lr_num]
        test_vis_dir = test_vis_dirs[lr_num]
        epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=test_vis_dir, period=1, trainable_params=trainable_params, visualise=False, testing=True)
        epoch_pred_cbs.append(epoch_pred_cb)

    return epoch_pred_cbs
