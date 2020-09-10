
import os
import numpy as np
from keras.optimizers import Adam, SGD

from callbacks import OptLearnerPredOnEpochEnd
from smpl_np import SMPLModel

from architectures.architecture_helpers import false_loss
from tools.data_helpers import format_offsetable_params, offset_params, format_distractor_dict, gather_input_data, gather_cb_data, format_joint_levels
from tools.model_helpers import emb_init_weights_np, construct_optlearner_model, freeze_layers, initialise_emb_layers, gather_optlearner_losses


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


def create_root_dir(run_id):
    exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/" + str(run_id) + "/"
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


def setup_test_dir(exp_dir, JOINT_LEVELS, LEARNING_RATES):
    # Gather experiment directory details
    model_name = get_exp_models(exp_dir)
    if JOINT_LEVELS is None:
        save_suffix = ""
    else:
        save_suffix = "_conditional"
    #save_suffix = "_non-zero_pose"

    # Create base testing directory
    model, logs_dir, control_logs_dir, test_vis_dir, control_dir = create_base_subdir(exp_dir, model_name, save_suffix)

    # Create subdirectories for the learning rates
    logs_dirs = []
    control_logs_dirs = []
    test_vis_dirs = []
    control_dirs = []
    for lr in LEARNING_RATES:
        sub_dir = "lr_{:02f}/".format(lr)
        logs_dir, control_logs_dir, test_vis_dir, control_dir = create_subdir(exp_dir, model_name, sub_dir, save_suffix)
        logs_dirs.append(logs_dir)
        control_logs_dirs.append(control_logs_dir)
        test_vis_dirs.append(test_vis_dir)
        control_dirs.append(control_dir)

    return model, logs_dirs, control_logs_dirs, test_vis_dirs, control_dirs


def setup_multinet_test_dir(run_id, exp_dirs, LEARNING_RATES):
    # Gather experiment directory details
    models = []
    for exp_dir in exp_dirs:
        model_name = get_exp_models(exp_dir)
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


def setup_test_data(trainable_params, DISTRACTOR, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, JOINT_LEVELS, data_samples, num_test_samples=5, num_cb_samples=5, MODE="RODRIGUES", LOAD_DATA_DIR=None):
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
    X_test, Y_test = gather_input_data(data_samples, smpl, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, param_trainable, num_test_samples=num_test_samples, MODE=MODE, LOAD_DATA_DIR=LOAD_DATA_DIR)

    # Render silhouettes for the callback data
    X_cb, Y_cb, silh_cb = gather_cb_data(X_test, Y_test, data_samples, num_cb_samples, where="front")

    return X_test, Y_test, X_cb, Y_cb, silh_cb, smpl, kin_tree, trainable_params, param_trainable, DISTRACTOR


def setup_embedding_weights(data_samples, X_params, DISTRACTOR, param_trainable, DIST="uniform"):
    # Generate initial embedding layer weights
    initial_weights = np.zeros((data_samples, 85))
    emb_initialiser = emb_init_weights_np(X_params, distractor=DISTRACTOR, dist=DIST)
    for param_name, trainable in param_trainable.items():
        param_number = int(param_name[6:8])
        emb_init_ = emb_initialiser(param=param_number, offset=trainable)
        initial_weights[:, param_number] = emb_init_(shape=(data_samples,))

    return emb_initialiser, initial_weights


def setup_test_model(model, ARCHITECTURE, data_samples, param_trainable, emb_initialiser, initial_weights, INPUT_TYPE, TRAIN_LR, groups):
    # Retrieve model architecture and load weights
    optlearner_model = construct_optlearner_model(ARCHITECTURE, param_trainable, emb_initialiser, data_samples, INPUT_TYPE, groups)
    optlearner_model.load_weights(model)

    # Freeze all layers except for the required embedding layers
    optlearner_model = freeze_layers(optlearner_model, param_trainable)

    # Set the weights of the embedding layers
    optlearner_model = initialise_emb_layers(optlearner_model, param_trainable, initial_weights)

    # Compile the model
    optimizer = Adam(lr=TRAIN_LR, decay=0.0)
    #optimizer = SGD(lr=TRAIN_LR, momentum=0.0, nesterov=False)
    optlearner_loss, optlearner_loss_weights = gather_optlearner_losses(INPUT_TYPE, ARCHITECTURE)
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
