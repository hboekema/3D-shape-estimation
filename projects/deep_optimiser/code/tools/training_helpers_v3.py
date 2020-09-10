
import os

import numpy as np
import pickle
import json
from keras.callbacks import LambdaCallback, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam, SGD

from smpl_np import SMPLModel
from silhouette_generator_v2 import OptLearnerUpdateGenerator
from callbacks_v2 import PredOnEpochEnd, OptLearnerPredOnEpochEnd, OptimisationCallback, OptLearnerLossGraphCallback, GeneratorParamErrorCallback
from tools.data_helpers_v2 import format_offsetable_params, offset_params, format_distractor_dict, gather_input_data, gather_cb_data, format_joint_levels, get_new_weights, save_dist_info
from tools.model_helpers_v2 import construct_optlearner_model, gather_optlearner_losses, gather_optlearner_summed_losses
from architectures.architecture_helpers import false_loss, no_loss
from architectures.initialisers.simple_initialiser_v2 import emb_init_weights


def update_weights_wrapper(DISTRACTOR, num_samples, PERIOD, trainable_params, optlearner_model, gt_params=None, generator=None, offset_nt={}, dist="uniform", reset_to_zero=False, log_path=None):
    """ Wrapper for weight update functionality """
    recognised_modes = ["uniform", "gaussian", "normal"]
    assert dist in recognised_modes, "distribution not recognised"
    assert generator is not None or gt_params is not None, "generator or GT parameters must be passed to update weights function"

    def update_weights(epoch, logs):
        # Update a block of parameters
        BL_SIZE = num_samples // PERIOD
        #print("epoch: " + str(epoch))
        BL_INDEX = epoch % PERIOD
        #print("BL_SIZE: " + str(BL_SIZE))
        #print("BL_INDEX: " + str(BL_INDEX))

        if generator is not None:
            gt_data, _ = generator.yield_data(epoch, render=False)
            gt_params = gt_data[1]
            #print("Gt params shape: " + str(gt_params.shape))
            #print("\n------------------\n" + str(gt_params[0]) + "\n------------------\n")

        updated_weights = np.zeros((num_samples, 85))
        new_weights = get_new_weights(DISTRACTOR, trainable_params, gt_params, offset_nt, dist, reset_to_zero, BL_INDEX, BL_SIZE)
        for param in range(85):
            param_id = "param_{:02d}".format(param)
            layer = optlearner_model.get_layer(param_id)
            weights = np.array(layer.get_weights())
            #print(weights)
            #print('weights shape: '+str(weights.shape))
            #exit(1)
            updated_weights[:, param] = np.squeeze(weights)
            weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] = np.reshape(new_weights[:, :, param], (1, BL_SIZE, 1))
            layer.set_weights(weights)


        if log_path is not None:
            preds_path = log_path + "preds_dist.txt"
            gt_path = log_path + "gt_dist.txt"
            diff_path = log_path + "diff_dist.txt"
            delta_weights = gt_params - updated_weights

            save_dist_info(preds_path, updated_weights, epoch)
            save_dist_info(gt_path, gt_params, epoch)
            save_dist_info(diff_path, delta_weights, epoch)

    return update_weights


def setup_train_dir(run_id):
    # Create experiment directory
    exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/" + str(run_id) + "/"
    #exp_dir = "/scratches/robot_2/hjhb2/projects/deep_optimiser/experiments/" + str(run_id) + "/"
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

    return exp_dir, model_dir, logs_dir, opt_logs_dir, train_vis_dir, test_vis_dir, opt_vis_dir, code_dir, tensorboard_logs_dir


def print_summary_wrapper(path):
    def print_summary(s):
        with open(path + "model_summary.txt",'a') as f:
            f.write(s)
        print(s)
    return print_summary


def setup_train_data(trainable_params, DISTRACTOR, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, data_samples, BATCH_SIZE, train_vis_dir, num_cb_samples=5, MODE="RODRIGUES", LOAD_DATA_DIR=None, use_generator=False, RESET_PERIOD=None, groups=None, TRAIN_PERIOD=5, OFFSET_NT={}, dist="uniform"):
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

    # Gather non-trainable offsets
    OFFSET_NT = format_distractor_dict(OFFSET_NT, PARAMS_TO_OFFSET)

    # Define the kinematic tree
    kin_tree = format_joint_levels(groups)
    #print(kin_tree)
    efficient_kin_tree = [[param for param in level if param in trainable_params] for level in kin_tree]
    efficient_kin_tree = [entry for entry in efficient_kin_tree if entry]   # filter empty entries
    #print(efficient_kin_tree)

    # Generate the data from the SMPL parameters
    print("loading SMPL...")
    smpl = SMPLModel('/data/cvfs/hjhb2/projects/deep_optimiser/code/keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    # Generate and format the data
    print("Gather input data...")
    if use_generator:
        #print(param_trainable.keys())
        #exit(1)
        trainable_params_mask = [int(param_trainable[key]) for key in sorted(param_trainable.keys(), key=lambda x: int(x.replace("param_", "")))]
        update_generator = OptLearnerUpdateGenerator(data_samples, RESET_PERIOD, POSE_OFFSET, PARAMS_TO_OFFSET, ARCHITECTURE, batch_size=BATCH_SIZE, smpl=smpl, shuffle=True, save_path=train_vis_dir, trainable_params_mask=trainable_params_mask, kin_tree=efficient_kin_tree, train_period=TRAIN_PERIOD, dist=dist)
        X_train, Y_train = update_generator.yield_data()
        print("Y_train shapes: " + str([datum.shape for datum in Y_train]))
    else:
        update_generator = None
        X_train, Y_train = gather_input_data(data_samples, smpl, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, param_trainable, num_test_samples=-1, MODE=MODE, LOAD_DATA_DIR=LOAD_DATA_DIR, use_generator=use_generator, kin_tree=efficient_kin_tree, dist=dist)

    # Render silhouettes for the callback data
    X_cb, Y_cb, silh_cb = gather_cb_data(X_train, Y_train, data_samples, num_cb_samples, where="spread")

    return X_train, Y_train, X_cb, Y_cb, silh_cb, smpl, trainable_params, param_trainable, DISTRACTOR, POSE_OFFSET, update_generator, OFFSET_NT


def setup_emb_initialiser(X_params, DISTRACTOR, trainable_params, OFFSET_NT={}, RESET_PRED_TO_ZERO=False, DIST="uniform"):
    X_params = np.array(X_params)
    initial_weights = get_new_weights(DISTRACTOR, trainable_params, X_params, offset_nt=OFFSET_NT, dist=DIST, reset_to_zero=RESET_PRED_TO_ZERO, BL_INDEX=0, BL_SIZE=X_params.shape[0])
    emb_initialiser = emb_init_weights(initial_weights)

    return emb_initialiser


def setup_train_model(ARCHITECTURE, data_samples, param_trainable, emb_initialiser, INPUT_TYPE, LR, LOSSES_WEIGHTS, GROUPS, UPDATE_WEIGHT, optimizer, BATCH_SIZE, DROPOUT, model_path=None, INCLUDE_SHAPE=True):
    # Retrieve model architecture and load weights
    if ARCHITECTURE == "GroupedConv2DOptLearnerArchitecture" and model_path is not None:
        # Load in pre-trained Conv2D network and fix its weights
        param_2D_trainable = False    # if model is given, parameters for network are fixed
    else:
        param_2D_trainable = True

    optlearner_model = construct_optlearner_model(ARCHITECTURE, param_trainable, emb_initialiser, data_samples, INPUT_TYPE, GROUPS, UPDATE_WEIGHT, DROPOUT, param_2D_trainable, INCLUDE_SHAPE)

    if ARCHITECTURE == "GroupedConv2DOptLearnerArchitecture" and model_path is not None:
        # Load in pre-trained Conv2D network and fix its weights
        optlearner_model.load_weights(model_path)

    # Compile the model
    if optimizer == "Adam":
        optimizer = Adam(lr=LR, decay=0.0)
        #optimizer = Adam(lr=LR, decay=0.0, clipvalue=100.0)
        #optimizer = Adam(lr=LR, decay=0.1)
        optlearner_loss, optlearner_loss_weights = gather_optlearner_losses(INPUT_TYPE, ARCHITECTURE, LOSSES_WEIGHTS)
        #optlearner_loss, optlearner_loss_weights = gather_optlearner_summed_losses(INPUT_TYPE, ARCHITECTURE, LOSSES_WEIGHTS)
    elif optimizer == "SGD":
        #optimizer = SGD(lr=LR, momentum=0.0, nesterov=False)
        #optimizer = SGD(lr=LR, momentum=0.0, nesterov=False, clipvalue=1.0)   # added clipvalue for 2D network
        optimizer = SGD(lr=LR, momentum=0.0, nesterov=False, clipvalue=100.0)   # added clipvalue for shape
        optlearner_loss, optlearner_loss_weights = gather_optlearner_losses(INPUT_TYPE, ARCHITECTURE, LOSSES_WEIGHTS, BATCH_SIZE)
    optlearner_model.compile(
            optimizer=optimizer,
            loss=optlearner_loss,
            loss_weights=optlearner_loss_weights,
            metrics={"delta_d_hat_sin_output": false_loss},
            #options=run_options,
            #run_metadata=run_metadata,
            )

    return optlearner_model, optlearner_loss, optlearner_loss_weights


def setup_train_cb(model_dir, MODEL_SAVE_PERIOD, USE_GENERATOR, logs_dir, smpl, X_cb, silh_cb, train_vis_dir, PREDICTION_PERIOD, trainable_params, RESET_PERIOD, data_samples, DISTRACTOR, optlearner_model, update_generator, OFFSET_NT, POSE_OFFSET, exp_dir, DIST, RESET_PRED_TO_ZERO, ARCHITECTURE, losses, loss_weights):
    # Callback functions
    # Create a model checkpoint after every few epochs
    model_save_checkpoint = ModelCheckpoint(
        model_dir + "model.{epoch:02d}-{delta_d_hat_mse_loss:.4f}.hdf5",
        monitor='loss', verbose=1, save_best_only=False, mode='auto',
        period=MODEL_SAVE_PERIOD, save_weights_only=True)

    losses_and_weights = zip(losses, loss_weights)
    active_losses_and_weights = [(i, loss_weight[1]) for i, loss_weight in enumerate(losses_and_weights) if loss_weight[0] != no_loss]
    active_losses, loss_weights = zip(*active_losses_and_weights)

    # Predict on sample params at the end of every few epochs
    if USE_GENERATOR:
        epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=False, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples, train_gen=train_vis_dir, ARCHITECTURE=ARCHITECTURE, losses=active_losses, loss_weights=loss_weights, generator=update_generator)
    else:
        epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=False, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples, ARCHITECTURE=ARCHITECTURE, losses=active_losses, loss_weights=loss_weights)
        #epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=True, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples)

    # Callback for distractor unit
    if USE_GENERATOR:
        weight_cb_wrapper = update_weights_wrapper(DISTRACTOR, data_samples, RESET_PERIOD, trainable_params, optlearner_model, generator=update_generator, offset_nt=OFFSET_NT, dist=DIST, reset_to_zero=RESET_PRED_TO_ZERO, log_path=logs_dir)
    else:
        weight_cb_wrapper = update_weights_wrapper(DISTRACTOR, data_samples, RESET_PERIOD, trainable_params, optlearner_model, offset_nt=OFFSET_NT, dist=DIST, reset_to_zero=RESET_PRED_TO_ZERO, log_path=logs_dir)
    #weight_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: weight_cb_wrapper(epoch, logs))
    weight_cb = LambdaCallback(on_epoch_begin=lambda epoch, logs: weight_cb_wrapper(epoch, logs))

    # Callback for loss plotting during training
    plotting_cb = OptLearnerLossGraphCallback(exp_dir, graphing_period=100)

    # Collect callbacks
    callbacks = [model_save_checkpoint, epoch_pred_cb, weight_cb, plotting_cb]

    # Parameter error callback
    if USE_GENERATOR:
        param_error_cb = GeneratorParamErrorCallback(exp_dir, update_generator, PREDICTION_PERIOD, ARCHITECTURE)
        callbacks.append(param_error_cb)

    return callbacks


def train_model(setup_params, run_id):
    # Basic experimental setup
    RESET_PERIOD = setup_params["BASIC"]["RESET_PERIOD"]
    MODEL_SAVE_PERIOD = setup_params["BASIC"]["MODEL_SAVE_PERIOD"]
    PREDICTION_PERIOD = setup_params["BASIC"]["PREDICTION_PERIOD"]
    OPTIMISATION_PERIOD = setup_params["BASIC"]["OPTIMISATION_PERIOD"]
    MODE = setup_params["BASIC"]["ROT_MODE"]
    DISTRACTOR = setup_params["BASIC"]["DISTRACTOR"]
    data_samples = setup_params["BASIC"]["NUM_SAMPLES"]
    NUM_CB_SAMPLES = setup_params["BASIC"]["NUM_CB_SAMPLES"]

    # Parameters setup
    trainable_params = setup_params["PARAMS"]["TRAINABLE"]

    # Data setup
    DATA_LOAD_DIR = setup_params["DATA"]["TRAIN_DATA_DIR"]
    POSE_OFFSET = setup_params["DATA"]["POSE_OFFSET"]
    OFFSET_NT = setup_params["DATA"]["OFFSET_NT"]
    RESET_PRED_TO_ZERO = setup_params["DATA"]["RESET_PRED_TO_ZERO"]
    PARAMS_TO_OFFSET = setup_params["DATA"]["PARAMS_TO_OFFSET"]
    USE_GENERATOR = setup_params["DATA"]["USE_GENERATOR"]
    DIST = setup_params["DATA"]["DIST"]
    INCLUDE_SHAPE = setup_params["DATA"]["INCLUDE_SHAPE"]

    # Model setup
    ARCHITECTURE = setup_params["MODEL"]["ARCHITECTURE"]
    BATCH_SIZE = setup_params["MODEL"]["BATCH_SIZE"]
    EPOCHS = setup_params["MODEL"]["EPOCHS"]
    INPUT_TYPE = setup_params["MODEL"]["INPUT_TYPE"]
    learning_rate = setup_params["MODEL"]["LEARNING_RATE"]
    DELTA_D_LOSS_WEIGHT = setup_params["MODEL"]["DELTA_D_LOSS_WEIGHT"]
    PC_LOSS_WEIGHT = setup_params["MODEL"]["PC_LOSS_WEIGHT"]
    DELTA_D_HAT_LOSS_WEIGHT = setup_params["MODEL"]["DELTA_D_HAT_LOSS_WEIGHT"]
    UPDATE_LOSS_WEIGHT = setup_params["MODEL"]["UPDATE_LOSS_WEIGHT"]
    CONV2D_LOSS_WEIGHT = setup_params["MODEL"]["CONV2D_LOSS_WEIGHT"]
    CONV2D_PRETRAINED = setup_params["MODEL"]["CONV2D_PRETRAINED"]
    if CONV2D_PRETRAINED:
        CONV2D_LOSS_WEIGHT = 0.0
    LOSSES_WEIGHTS = [DELTA_D_LOSS_WEIGHT, PC_LOSS_WEIGHT, DELTA_D_HAT_LOSS_WEIGHT, UPDATE_LOSS_WEIGHT, CONV2D_LOSS_WEIGHT]
    TRAIN_PERIOD = setup_params["MODEL"]["TRAIN_PERIOD"]
    optimizer = setup_params["MODEL"]["OPTIMIZER"]
    DROPOUT = setup_params["MODEL"]["DROPOUT"]

    # Test setup
    groups = setup_params["TEST"]["joint_levels"]
    test_lrs = setup_params["TEST"]["learning_rates"]
    update_weight = test_lrs[0]


    """ Directory set-up """

    exp_dir, model_dir, logs_dir, opt_logs_dir, train_vis_dir, test_vis_dir, opt_vis_dir, code_dir, tensorboard_logs_dir = setup_train_dir(run_id)


    """ Data set-up """

    # Parameter setup
    X_train, Y_train, X_cb, Y_cb, silh_cb, smpl, trainable_params, param_trainable, DISTRACTOR, POSE_OFFSET, update_generator, OFFSET_NT = setup_train_data(trainable_params, DISTRACTOR, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, data_samples, BATCH_SIZE, train_vis_dir, num_cb_samples=NUM_CB_SAMPLES, MODE=MODE, LOAD_DATA_DIR=DATA_LOAD_DIR, use_generator=USE_GENERATOR, RESET_PERIOD=RESET_PERIOD, groups=groups, TRAIN_PERIOD=TRAIN_PERIOD, OFFSET_NT=OFFSET_NT, dist=DIST)


    """ Model set-up """

    # Set-up the embedding initialiser
    emb_initialiser = setup_emb_initialiser(X_train[1], DISTRACTOR, trainable_params, OFFSET_NT=OFFSET_NT, RESET_PRED_TO_ZERO=RESET_PRED_TO_ZERO, DIST=DIST)

    # Set-up the model
    optlearner_model, losses, loss_weights = setup_train_model(ARCHITECTURE, data_samples, param_trainable, emb_initialiser, INPUT_TYPE, learning_rate, LOSSES_WEIGHTS, groups, update_weight, optimizer, BATCH_SIZE, DROPOUT, model_path=CONV2D_PRETRAINED, INCLUDE_SHAPE=INCLUDE_SHAPE)
    optlearner_model.summary()

    """ Callback set-up """

    callbacks = setup_train_cb(model_dir, MODEL_SAVE_PERIOD, USE_GENERATOR, logs_dir, smpl, X_cb, silh_cb, train_vis_dir, PREDICTION_PERIOD, trainable_params, RESET_PERIOD, data_samples, DISTRACTOR, optlearner_model, update_generator, OFFSET_NT, POSE_OFFSET, exp_dir, DIST, RESET_PRED_TO_ZERO, ARCHITECTURE, losses, loss_weights)


    """ Training loop"""

    # Run the main loop
    if USE_GENERATOR:
        optlearner_model.fit_generator(
                update_generator,
                steps_per_epoch=data_samples//BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=False,
                workers=1,
                max_queue_size=10
                )
    else:
        optlearner_model.fit(
                x=X_data,
                y=Y_data,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                #steps_per_epoch=steps_per_epoch,
                #validation_data=(X_val, Y_val),
                #validation_steps=Y_val,
                callbacks=callbacks,
                )


    # Store the model
    print("Saving model to " + str(model_dir) + "model.final.hdf5...")
    optlearner_model.save_weights(model_dir + "model.final.hdf5")

    return optlearner_model

