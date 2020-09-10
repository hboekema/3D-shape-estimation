
import os
import numpy as np
from keras.optimizers import Adam, SGD

from callbacks_v2 import OptLearnerPredOnEpochEnd, TestingCallback
from smpl_np import SMPLModel

from architectures.architecture_helpers import false_loss
from architectures.initialisers.simple_initialiser_v2 import emb_init_weights
from tools.data_helpers_v2 import format_offsetable_params, format_distractor_dict, gather_input_data, gather_cb_data, format_joint_levels, get_new_weights, save_dist_info
from tools.model_helpers_v2 import construct_optlearner_model, freeze_layers, initialise_emb_layers, gather_test_optlearner_losses


""" Directory operations """

def gen_save_suffix(JOINT_LEVELS, DIST, RESET_PRED_TO_ZERO, is_train_dist):
    save_suffix = ""
    if is_train_dist:
        save_suffix += "_train"
    else:
        save_suffix += "_test"
    save_suffix = save_suffix + "_" + DIST
    if RESET_PRED_TO_ZERO:
        save_suffix += "_zero_init"
    else:
        save_suffix += "_random_init"
    if JOINT_LEVELS is not None and len(JOINT_LEVELS) > 1:
        save_suffix += "_conditional"
    return save_suffix


def get_exp_models(exp_dir):
    print("Experiment directory: " + str(exp_dir))
    models = os.listdir(exp_dir + "models/")
    if len(models) == 0:
        print("No models for this experiment. Exiting.")
        exit(1)
    else:
        models.sort(key=lambda x: float(x[x.find("-")+1 : x.find(".hdf5")]))
        best_model = models[0]
        models.sort(key=lambda x: int(x[x.find("model.")+6 : x.find("-")]))
        latest_model = models[-1]

        model_names = np.unique([best_model, latest_model])
        print("Using models '{}'".format(model_names))

    return model_names


def create_root_dir(run_id):
    exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/" + str(run_id) + "/"
    #exp_dir = "/scratches/robot_2/hjhb2/projects/deep_optimiser/experiments/" + str(run_id) + "/"
    logs_dir = exp_dir + "logs/"
    test_vis_dir = exp_dir + "test_vis/"
    os.mkdir(exp_dir)
    os.mkdir(logs_dir)
    os.mkdir(test_vis_dir)
    print("Experiment directory: \n" + str(exp_dir))

    return exp_dir, logs_dir, test_vis_dir


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


def initialise_pred_cbs(logs_dirs, test_vis_dirs, smpl, X_cb, silh_cb, trainable_params, learning_rates, ARCHITECTURE):
    assert len(logs_dirs) == len(learning_rates)
    assert len(test_vis_dirs) == len(learning_rates)

    input_names = ["embedding_index", "gt_params", "gt_pc", "gt_silh"]
    if ARCHITECTURE == "PeriodicOptLearnerArchitecture":
        input_names.append("params_to_train")
    if ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture":
        input_names.append("trainable_params")

    test_logging_cbs = []
    epoch_pred_cbs = []
    for lr_num, lr in enumerate(learning_rates):
        logs_dir = logs_dirs[lr_num]
        test_vis_dir = test_vis_dirs[lr_num]

        epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=test_vis_dir, period=1, trainable_params=trainable_params, visualise=False, testing=True, ARCHITECTURE=ARCHITECTURE)
        epoch_pred_cbs.append(epoch_pred_cb)

        test_logging_cb = TestingCallback(test_vis_dir, input_names=input_names)
        test_logging_cbs.append(test_logging_cb)

    return epoch_pred_cbs, test_logging_cbs


def create_test_dir_for_model(exp_dir, model_name, save_suffix, LEARNING_RATES, points_lr_weights=None, normals_lr_weights=None, OPTIMIZER="SGD"):
    # Create base testing directory
    model, logs_dir, control_logs_dir, test_vis_model_dir, control_dir = create_base_subdir(exp_dir, model_name, save_suffix)

    # Create subdirectories for the learning rates
    if normals_lr_weights is None:
        normals_lr_weights = [0.0 for entry in LEARNING_RATES]
    if points_lr_weights is None:
        points_lr_weights = [0.0 for entry in LEARNING_RATES]
    logs_dirs = []
    control_logs_dirs = []
    test_vis_dirs = []
    control_dirs = []
    for lr_num, lr in enumerate(LEARNING_RATES):
        sub_dir = OPTIMIZER
        if lr != 0.0:
            sub_dir += "_UpdateLR{:02f}".format(lr)
        elif normals_lr_weights[lr_num] != 0.0:
            sub_dir += "_NormLR{:02f}".format(normals_lr_weights[lr_num])
        elif points_lr_weights[lr_num] != 0.0:
            sub_dir += "_PointsLR{:02f}".format(points_lr_weights[lr_num])
        else:
            assert False, "all test weights zero"
        sub_dir += "/"

        logs_dir, control_logs_dir, test_vis_dir, control_dir = create_subdir(exp_dir, model_name, sub_dir, save_suffix)
        logs_dirs.append(logs_dir)
        control_logs_dirs.append(control_logs_dir)
        test_vis_dirs.append(test_vis_dir)
        control_dirs.append(control_dir)

    return model, logs_dirs, control_logs_dirs, test_vis_dirs, control_dirs, test_vis_model_dir


def setup_multinet_test_dir(run_id, exp_dirs, LEARNING_RATES):
    # Gather experiment directory details
    models = []
    for exp_dir in exp_dirs:
        model_name = get_exp_models(exp_dir)[0]
        model = exp_dir + "models/" + model_name
        models.append(model)


    # Create root experiment directory
    root_dir, _, _ = create_root_dir(run_id)

    # Create base testing directory
    model_name = "model"
    _, logs_dir, control_logs_dir, test_vis_dir, control_dir = create_base_subdir(root_dir, model_name)

    # Create subdirectories for the learning rates
    logs_dirs = []
    control_logs_dirs = []
    test_vis_dirs = []
    control_dirs = []
    for lr in LEARNING_RATES:
        sub_dir = "lr_{:02f}/".format(lr)
        logs_dir, control_logs_dir, test_vis_dir, control_dir = create_subdir(root_dir, model_name, sub_dir)
        logs_dirs.append(logs_dir)
        control_logs_dirs.append(control_logs_dir)
        test_vis_dirs.append(test_vis_dir)
        control_dirs.append(control_dir)

    return models, logs_dirs, control_logs_dirs, test_vis_dirs, control_dirs


def setup_test_data(trainable_params, DISTRACTOR, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, JOINT_LEVELS, data_samples, num_test_samples=5, num_cb_samples=5, MODE="RODRIGUES", LOAD_DATA_DIR=None, dist="uniform"):
    # Gather trainable params
    param_ids = ["param_{:02d}".format(i) for i in range(85)]
    trainable_params = format_offsetable_params(trainable_params)
    param_trainable = { param: (param in trainable_params) for param in param_ids }
    DISTRACTOR = format_distractor_dict(DISTRACTOR, trainable_params)

    # Gather offsetable params
    if PARAMS_TO_OFFSET == "trainable_params":
        PARAMS_TO_OFFSET = trainable_params
    PARAMS_TO_OFFSET = format_offsetable_params(PARAMS_TO_OFFSET)
    POSE_OFFSET = format_distractor_dict(POSE_OFFSET, PARAMS_TO_OFFSET)

    # Define the kinematic tree
    kin_tree = format_joint_levels(JOINT_LEVELS)

    # Generate the data from the SMPL parameters
    print("loading SMPL...")
    smpl = SMPLModel('/data/cvfs/hjhb2/projects/deep_optimiser/code/keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    # Generate and format the data
    print("Gather input data...")
    X_test, Y_test = gather_input_data(data_samples, smpl, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, param_trainable, num_test_samples=num_test_samples, MODE=MODE, LOAD_DATA_DIR=LOAD_DATA_DIR, kin_tree=kin_tree, dist=dist)

    # Render silhouettes for the callback data
    X_cb, Y_cb, silh_cb = gather_cb_data(X_test, Y_test, data_samples, num_cb_samples, where="front")

    return X_test, Y_test, X_cb, Y_cb, silh_cb, smpl, kin_tree, trainable_params, param_trainable, DISTRACTOR, POSE_OFFSET


def setup_embedding_weights(X_params, DISTRACTOR, trainable_params, DIST="uniform", OFFSET_NT={}, RESET_PRED_TO_ZERO=False, log_path=None):
    # Generate initial embedding layer weights
    X_params = np.array(X_params)
    initial_weights = get_new_weights(DISTRACTOR, trainable_params, X_params, offset_nt=OFFSET_NT, dist=DIST, reset_to_zero=RESET_PRED_TO_ZERO, BL_INDEX=0, BL_SIZE=X_params.shape[0])
    emb_initialiser = emb_init_weights(initial_weights)
    print("initial_weights shape: " + str(initial_weights.shape))
    initial_weights = np.reshape(initial_weights, (-1, 85))
    print("initial_weights shape: " + str(initial_weights.shape))

    if log_path is not None:
        preds_path = log_path + "preds_dist.txt"
        gt_path = log_path + "gt_dist.txt"
        diff_path = log_path + "diff_dist.txt"
        delta_weights = X_params - initial_weights

        save_dist_info(preds_path, initial_weights, 0)
        save_dist_info(gt_path, X_params, 0)
        save_dist_info(diff_path, delta_weights, 0)

    return emb_initialiser, initial_weights


def get_test_type(lr, optimizer):
    test_type = ""
    if lr != 0.0:
        test_type = "deep_opt"
    else:
        test_type = str(optimizer).lower() + "_gd"

    return test_type


def setup_test_model(model, ARCHITECTURE, data_samples, param_trainable, emb_initialiser, initial_weights, INPUT_TYPE, groups, update_weight, PREDS_WEIGHT, POINTS_GD_WEIGHT, NORMALS_GD_WEIGHT, OPTIMIZER, INCLUDE_SHAPE):
    # Retrieve model architecture and load weights
    optlearner_model = construct_optlearner_model(ARCHITECTURE, param_trainable, emb_initialiser, data_samples, INPUT_TYPE, groups, update_weight, INCLUDE_SHAPE=INCLUDE_SHAPE)
    optlearner_model.load_weights(model)

    # Freeze all layers except for the required embedding layers
    optlearner_model = freeze_layers(optlearner_model, param_trainable)

    # Set the weights of the embedding layers
    optlearner_model = initialise_emb_layers(optlearner_model, param_trainable, initial_weights)

    # Compile the model
    WEIGHTS = [POINTS_GD_WEIGHT, NORMALS_GD_WEIGHT, PREDS_WEIGHT]
    lr_value = 1.0
    if len([entry for entry in WEIGHTS if entry != 0.0]):
        lr_value = np.max(WEIGHTS)
        WEIGHTS = np.array(WEIGHTS)/lr_value
    else:
        assert False, "this test implementation takes exactly one non-zero loss"
    if OPTIMIZER == "SGD":
        optimizer = SGD(lr=lr_value, momentum=0.0, nesterov=False)
    elif OPTIMIZER == "Adam":
        optimizer = Adam(lr=lr_value, decay=0.0)
    else:
        assert False, "optimizer not recognised"

    optlearner_loss, optlearner_loss_weights = gather_test_optlearner_losses(ARCHITECTURE, [POINTS_GD_WEIGHT, NORMALS_GD_WEIGHT, PREDS_WEIGHT])
    optlearner_model.compile(
            optimizer=optimizer,
            loss=optlearner_loss,
            loss_weights=optlearner_loss_weights,
            metrics={"delta_d_hat_sin_output": false_loss},
            #options=run_options,
            #run_metadata=run_metadata,
            )

    # Print model summary
    optlearner_model.summary()

    return optlearner_model

