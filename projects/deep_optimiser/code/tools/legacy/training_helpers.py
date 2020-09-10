
import os

import numpy as np
from keras.callbacks import LambdaCallback, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam, SGD

from smpl_np import SMPLModel
from silhouette_generator import OptLearnerUpdateGenerator
from callbacks import PredOnEpochEnd, OptLearnerPredOnEpochEnd, OptimisationCallback, OptLearnerLossGraphCallback, GeneratorParamErrorCallback
from tools.data_helpers import format_offsetable_params, offset_params, format_distractor_dict, gather_input_data, gather_cb_data, format_joint_levels
from tools.model_helpers import construct_optlearner_model, gather_optlearner_losses
from architectures.architecture_helpers import false_loss, emb_init_weights


def update_weights_wrapper(DISTRACTOR, num_samples, PERIOD, trainable_params, optlearner_model, gt_data=None, generator=None, offset_nt=False, pose_offset={}, dist="uniform", reset_to_zero=False):
    """ Wrapper for weight update functionality """
    recognised_modes = ["uniform", "gaussian", "normal"]
    assert dist in recognised_modes
    epsilon = 1e-5

    # Set the distractor value
    if isinstance(DISTRACTOR, (int, float)):
        k = {param: DISTRACTOR for param in trainable_params}
    else:
        # k must be a dict with an entry for each variable parameter
        k = DISTRACTOR
    #print(k)
    #print(pose_offset)
    #exit(1)

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
            param_ids = ["param_{:02d}".format(i) for i in range(85)]
            not_trainable = [param for param in param_ids if param not in trainable_params]
            for param in param_ids:
                layer = optlearner_model.get_layer(param)
                weights = np.array(layer.get_weights())
                #print('weights shape: '+str(weights.shape))
                #exit(1)
                #if False:
                if not reset_to_zero:
                    weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] = gt_params[BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE, int(param[6:8])].reshape((1, BL_SIZE, 1))
                    if offset_nt and param in pose_offset.keys() and param in not_trainable:
                        if dist == "gaussian" or dist == "normal":
                            extra_offset = [ np.random.normal(loc=0.0, scale=pose_offset[param], size=(BL_SIZE, weights[i].shape[1])) for i in range(weights.shape[0])]
                        elif dist == "uniform":
                            extra_offset = [ (1-2*np.random.rand(BL_SIZE, weights[i].shape[1])) * pose_offset[param] for i in range(weights.shape[0])]
                        weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] += extra_offset
                else:
                    weights_new = np.array([np.zeros((BL_SIZE, weights[i].shape[1])) + epsilon for i in range(weights.shape[0])]).reshape(1, BL_SIZE, 1)
                    weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] = np.copy(weights_new)    # only update the required block of weights
                #print(weights)
                layer.set_weights(weights)

#        param_ids = ["param_{:02d}".format(i) for i in range(85)]
#        not_trainable = [param for param in param_ids if param not in trainable_params]
#        for param in not_trainable:
#            layer = optlearner_model.get_layer(param)
#            weights = np.array(layer.get_weights())
#            if offset_nt and param in pose_offset.keys():
#                offset = [ (1-2*np.random.rand(BL_SIZE, weights[i].shape[1])) * pose_offset[param] for i in range(weights.shape[0])]
#                weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] = offset
#            layer.set_weights(weights)

        for param in trainable_params:
            layer = optlearner_model.get_layer(param)
            weights = np.array(layer.get_weights())
            #print("weights " + str(weights))
            #print('weights shape: '+str(weights.shape))
            #exit(1)
            if False:
            #if not reset_to_zero:
                if dist == "gaussian" or dist == "normal":
                    weights_new = [ np.random.normal(loc=0.0, scale=k[param], size=(BL_SIZE, weights[i].shape[1])) for i in range(weights.shape[0])]
                elif dist == "uniform":
                    weights_new = [ (1-2*np.random.rand(BL_SIZE, weights[i].shape[1])) * k[param] for i in range(weights.shape[0])]
                #print("weights new " + str(weights_new))
                weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] += weights_new    # only update the required block of weights
            else:
                weights_new = np.array([np.zeros((BL_SIZE, weights[i].shape[1])) + epsilon for i in range(weights.shape[0])]).reshape(1, BL_SIZE, 1)
                weights[:, BL_INDEX*BL_SIZE:(BL_INDEX+1)*BL_SIZE] = np.copy(weights_new)    # only update the required block of weights
            #print(weights)
            layer.set_weights(weights)
            #exit(1)


        #all_weights = []
        #for param in param_ids:
        #    layer = optlearner_model.get_layer(param)
        #    weights = np.array(layer.get_weights())
        #    all_weights.append(weights[0])

        #all_weights = np.concatenate(all_weights, axis=-1)
        #print(all_weights)
        #exit(1)


    def update_weights_artificially(epoch, logs):
        """ DEPRECATED """
        for param in trainable_params:
            gt_weights = gt_data[:, int(param[6:8])]
            layer = optlearner_model.get_layer(param)
            weights = np.array(layer.get_weights())
            #print("weights " + str(weights))
            #print('weights shape: '+str(weights.shape))
            #exit(1)

            # Compute the offset from the ground truth value
            offset_value = np.random.uniform(low=-k[param], high=k[param], size=(num_samples, 1))
            #offset_value = K.random_uniform(shape=[num_samples], minval=-k[param], maxval=k[param], dtype="float32")
            block_size = num_samples // PERIOD
            means = [np.sqrt(float((i + 1 + epoch) % PERIOD)/PERIOD) for i in range(PERIOD)]
            np.random.shuffle(means)
            factors = np.concatenate([np.random.normal(loc=means[i], scale=0.01, size=(block_size, 1)) for i in range(PERIOD)])
            offset_value *= factors
            #print("offset_value shape: " + str(offset_value.shape))
            new_weights = gt_weights.reshape(offset_value.shape) + offset_value
            new_weights = new_weights.reshape(weights.shape)
            #print("new_weights shape: " + str(new_weights.shape))
            #factors = Concatenate()([K.random_normal(shape=[block_size], mean=means[i], stddev=0.01) for i in range(PERIOD)])
            #offset_value = Multiply()([offset_value, factors])
            #weights = Add()([gt_weights, offset_value])

            layer.set_weights(new_weights)
            #exit(1)

    if gt_data is None:
        return update_weights
    else:
        return update_weights_artificially


def setup_train_dir(run_id):
    # Create experiment directory
    exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/" + str(run_id) + "/"
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


def setup_train_data(trainable_params, DISTRACTOR, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, data_samples, BATCH_SIZE, train_vis_dir, num_cb_samples=5, MODE="RODRIGUES", LOAD_DATA_DIR=None, use_generator=False, RESET_PERIOD=None, groups=None, TRAIN_PERIOD=5):
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
        trainable_params_mask = [int(param_trainable[key]) for key in sorted(param_trainable.keys(), key=lambda x: int(x[6:8]))]
        update_generator = OptLearnerUpdateGenerator(data_samples, RESET_PERIOD, POSE_OFFSET, PARAMS_TO_OFFSET, ARCHITECTURE, batch_size=BATCH_SIZE, smpl=smpl, shuffle=True, save_path=train_vis_dir, trainable_params_mask=trainable_params_mask, kin_tree=efficient_kin_tree, train_period=TRAIN_PERIOD)
        X_train, Y_train = update_generator.yield_data()
        print("Y_train shapes: " + str([datum.shape for datum in Y_train]))
    else:
        update_generator = None
        X_train, Y_train = gather_input_data(data_samples, smpl, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, param_trainable, num_test_samples=-1, MODE=MODE, LOAD_DATA_DIR=LOAD_DATA_DIR, use_generator=use_generator, kin_tree=efficient_kin_tree)

    # Render silhouettes for the callback data
    X_cb, Y_cb, silh_cb = gather_cb_data(X_train, Y_train, data_samples, num_cb_samples, where="spread")

    return X_train, Y_train, X_cb, Y_cb, silh_cb, smpl, trainable_params, param_trainable, DISTRACTOR, POSE_OFFSET, update_generator


def setup_emb_initialiser(X_params, DISTRACTOR, period=None, OFFSET_NT=False, POSE_OFFSET=None, RESET_PRED_TO_ZERO=False, DIST="uniform"):
    if RESET_PRED_TO_ZERO:
        emb_initialiser = emb_init_weights(X_params, period=period, distractor=DISTRACTOR, dist=DIST, reset_to_zero=RESET_PRED_TO_ZERO)
    elif OFFSET_NT:
        emb_initialiser = emb_init_weights(X_params, period=period, distractor=DISTRACTOR, pose_offset=POSE_OFFSET, dist=DIST)
    else:
        emb_initialiser = emb_init_weights(X_params, period=period, distractor=DISTRACTOR, dist=DIST)

    return emb_initialiser


def setup_train_model(ARCHITECTURE, data_samples, param_trainable, emb_initialiser, INPUT_TYPE, LR, LOSSES_WEIGHTS, GROUPS, UPDATE_WEIGHT):
    # Retrieve model architecture and load weights
    optlearner_model = construct_optlearner_model(ARCHITECTURE, param_trainable, emb_initialiser, data_samples, INPUT_TYPE, GROUPS, UPDATE_WEIGHT)

    # Compile the model
    optimizer = Adam(lr=LR, decay=0.0)
    #optimizer = SGD(lr=TRAIN_LR, momentum=0.0, nesterov=False)
    optlearner_loss, optlearner_loss_weights = gather_optlearner_losses(INPUT_TYPE, ARCHITECTURE, LOSSES_WEIGHTS)
    optlearner_model.compile(
            optimizer=optimizer,
            loss=optlearner_loss,
            loss_weights=optlearner_loss_weights,
            metrics={"delta_d_hat_sin_output": false_loss},
            #options=run_options,
            #run_metadata=run_metadata,
            )

    return optlearner_model


def setup_train_cb(model_dir, MODEL_SAVE_PERIOD, USE_GENERATOR, logs_dir, smpl, X_cb, silh_cb, train_vis_dir, PREDICTION_PERIOD, trainable_params, RESET_PERIOD, data_samples, DISTRACTOR, optlearner_model, update_generator, OFFSET_NT, POSE_OFFSET, exp_dir, DIST, RESET_PRED_TO_ZERO, ARCHITECTURE):
    # Callback functions
    # Create a model checkpoint after every few epochs
    model_save_checkpoint = ModelCheckpoint(
        model_dir + "model.{epoch:02d}-{delta_d_hat_mse_loss:.4f}.hdf5",
        monitor='loss', verbose=1, save_best_only=False, mode='auto',
        period=MODEL_SAVE_PERIOD, save_weights_only=True)

    # Predict on sample params at the end of every few epochs
    if USE_GENERATOR:
        epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=False, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples, train_gen=train_vis_dir, ARCHITECTURE=ARCHITECTURE)
    else:
        epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=False, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples, ARCHITECTURE=ARCHITECTURE)
        #epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=train_vis_dir, period=PREDICTION_PERIOD, trainable_params=trainable_params, visualise=False, testing=True, RESET_PERIOD=RESET_PERIOD, data_samples=data_samples)

    # Callback for distractor unit
    if USE_GENERATOR:
        weight_cb_wrapper = update_weights_wrapper(DISTRACTOR, data_samples, RESET_PERIOD, trainable_params, optlearner_model, generator=update_generator, offset_nt=OFFSET_NT, pose_offset=POSE_OFFSET, dist=DIST, reset_to_zero=RESET_PRED_TO_ZERO)
    else:
        weight_cb_wrapper = update_weights_wrapper(DISTRACTOR, data_samples, RESET_PERIOD, trainable_params, optlearner_model, offset_nt=OFFSET_NT, pose_offset=POSE_OFFSET, dist=DIST, reset_to_zero=RESET_PRED_TO_ZERO)
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

    # Model setup
    ARCHITECTURE = setup_params["MODEL"]["ARCHITECTURE"]
    BATCH_SIZE = setup_params["MODEL"]["BATCH_SIZE"]
    EPOCHS = setup_params["MODEL"]["EPOCHS"]
    INPUT_TYPE = setup_params["MODEL"]["INPUT_TYPE"]
    learning_rate = setup_params["MODEL"]["LEARNING_RATE"]
    DELTA_D_LOSS_WEIGHT = setup_params["MODEL"]["DELTA_D_LOSS_WEIGHT"]
    PC_LOSS_WEIGHT = setup_params["MODEL"]["PC_LOSS_WEIGHT"]
    DELTA_D_HAT_LOSS_WEIGHT = setup_params["MODEL"]["DELTA_D_HAT_LOSS_WEIGHT"]
    LOSSES_WEIGHTS = [DELTA_D_LOSS_WEIGHT, PC_LOSS_WEIGHT, DELTA_D_HAT_LOSS_WEIGHT]
    TRAIN_PERIOD = setup_params["MODEL"]["TRAIN_PERIOD"]

    # Test setup
    groups = setup_params["TEST"]["joint_levels"]
    test_lrs = setup_params["TEST"]["learning_rates"]
    update_weight = test_lrs[0]


    """ Directory set-up """

    exp_dir, model_dir, logs_dir, opt_logs_dir, train_vis_dir, test_vis_dir, opt_vis_dir, code_dir, tensorboard_logs_dir = setup_train_dir(run_id)


    """ Data set-up """

    # Parameter setup
    X_train, Y_train, X_cb, Y_cb, silh_cb, smpl, trainable_params, param_trainable, DISTRACTOR, POSE_OFFSET, update_generator = setup_train_data(trainable_params, DISTRACTOR, PARAMS_TO_OFFSET, POSE_OFFSET, ARCHITECTURE, data_samples, BATCH_SIZE, train_vis_dir, num_cb_samples=NUM_CB_SAMPLES, MODE=MODE, LOAD_DATA_DIR=DATA_LOAD_DIR, use_generator=USE_GENERATOR, RESET_PERIOD=RESET_PERIOD, groups=groups, TRAIN_PERIOD=TRAIN_PERIOD)


    """ Model set-up """

    # Set-up the embedding initialiser
    emb_initialiser = setup_emb_initialiser(X_train[1], DISTRACTOR, period=None, OFFSET_NT=OFFSET_NT, POSE_OFFSET=POSE_OFFSET, RESET_PRED_TO_ZERO=RESET_PRED_TO_ZERO, DIST=DIST)

    # Set-up the model
    optlearner_model = setup_train_model(ARCHITECTURE, data_samples, param_trainable, emb_initialiser, INPUT_TYPE, learning_rate, LOSSES_WEIGHTS, groups, update_weight)
    optlearner_model.summary()

    """ Callback set-up """

    callbacks = setup_train_cb(model_dir, MODEL_SAVE_PERIOD, USE_GENERATOR, logs_dir, smpl, X_cb, silh_cb, train_vis_dir, PREDICTION_PERIOD, trainable_params, RESET_PERIOD, data_samples, DISTRACTOR, optlearner_model, update_generator, OFFSET_NT, POSE_OFFSET, exp_dir, DIST, RESET_PRED_TO_ZERO, ARCHITECTURE)


    """ Training loop"""

    # Run the main loop
    if USE_GENERATOR:
        optlearner_model.fit_generator(
                update_generator,
                steps_per_epoch=data_samples//BATCH_SIZE,
                epochs=EPOCHS,
                #max_queue_size=1,
                callbacks=callbacks,
                shuffle=True,
                use_multiprocessing=False,
                workers=1
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

