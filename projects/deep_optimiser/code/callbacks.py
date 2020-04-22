import os
import sys
import json
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from copy import copy

sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from render_mesh import Mesh
from smpl_np_rot_v6 import print_mesh, print_point_clouds


class OptLearnerPredOnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, log_path, smpl_model, train_inputs=[[None], [None], [None]], train_silh=None, val_inputs=[[None], [None], [None]], val_silh=None, test_inputs=[[None], [None], [None]], test_silh=None, pred_path="../", period=5, trainable_params=[], visualise=True, testing=False, RESET_PERIOD=10, data_samples=10000, train_gen=None, val_gen=None, test_gen=None):
        # Open the log files
        epoch_log_path = os.path.join(log_path, "losses.txt")
        self.epoch_log = open(epoch_log_path, mode='wt', buffering=1)

        params_mse_log_path = os.path.join(log_path, "params_mse.txt")
        os.system("touch " + params_mse_log_path)
        self.params_mse_log = open(params_mse_log_path, mode='wt', buffering=1)

        delta_d_log_path = os.path.join(log_path, "delta_d.txt")
        os.system("touch " + delta_d_log_path)
        self.delta_d_log = open(delta_d_log_path, mode='wt', buffering=1)

        # Model to use to create meshes from SMPL parameters
        self.smpl = smpl_model

        # Store path for prediction visualisations
        self.pred_path = pred_path

        # Store data to be used for examples
        self.input_data = {"train": train_inputs, "val": val_inputs, "test": test_inputs}
        self.gt_silhouettes = {"train": train_silh, "val": val_silh, "test": test_silh}
        self.generator_paths = {"train": train_gen, "val": val_gen, "test": test_gen}

        # Store the prediction and optimisation periods
        self.period = period

        # Store model and architecture
        self.model = None

        # Store trainable parameter names
        self.trainable_params = trainable_params

        # Test or train mode
        self.testing = testing

        # Reset values
        self.RESET_PERIOD = RESET_PERIOD
        self.data_samples = data_samples
        self.examples = np.array(train_inputs[0])


    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        """ Store initialisation values """
        if int(epoch) == 0:
            self.on_epoch_end(epoch=-1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        """ Store the model loss and accuracy at the end of every epoch, and store a model prediction on data """
        #print("Callback called at epoch " + str(epoch))
        epoch = int(epoch)
        if logs is not None:
            self.epoch_log.write(json.dumps({'epoch': epoch})[:-1] + ", " + json.dumps(logs)[1:] + '\n')

        #if (epoch + 1) % self.period == 0 or epoch == 0 or epoch == -1:
        if epoch % self.period == 0 or epoch == -1:
            # Predict on all of the given input parameters
            for data_type, data in self.input_data.items():
                if data[0][0] is not None or self.generator_paths[data_type] is not None:
                    print("Saving to directory: \n{}\n".format(self.pred_path))
                    # Predict on these input parameters
                    #print("data value: " + str(data))
                    gen_path = self.generator_paths[data_type]
                    if gen_path is not None:
                        gen_path = gen_path + "cb_samples_E{}.npz".format(epoch)
                        try:
                            with np.load(gen_path, allow_pickle=True) as temp_data:
                                print(temp_data.keys())
                                if "trainable_params" in temp_data.keys():
                                    data = [temp_data["indices"], temp_data["params"], temp_data["pcs"], temp_data["trainable_params"]]
                                else:
                                    data = [temp_data["indices"], temp_data["params"], temp_data["pcs"]]
                        except Exception as e:
                            print("Skipping - load failed with exception '{}'".format(e))
                            return None

                    data_dict = {"embedding_index": np.array(data[0]), "gt_params": np.array(data[1]), "gt_pc": np.array(data[2])}
                    if len(data) == 4:
                        data_dict["trainable_params"] = np.array(data[3])
                    preds = self.model.predict(data_dict) #, batch_size=len(data[0]))

                    print(str(data_type))
                    print("------------------------------------")

                    #metrics_names = self.model.metrics_names[:-2]
                    metrics_names = self.model.metrics_names[:-1]
                    #print(metrics_names)
                    output_names = [metric[:-5] for i, metric in enumerate(metrics_names) if i > 0]
                    preds_dict = {output_name: preds[i] for i, output_name in enumerate(output_names)}
                    #print(preds_dict)
                    #exit(1)

                    #print("GT SMPL for first example: " + str(data[1][0]))
                    #print("Diff for first example: " + str(data[1][0] - preds_dict["learned_params"][0]))

                    self.delta_d_log.write('epoch {:05d}\n'.format(epoch + 1))
                    param_diff_sines = np.abs(np.sin(0.5*(data[1] - preds_dict["learned_params"])))
                    delta_d_diff_sines = np.abs(np.sin(0.5*(preds_dict["delta_d"] - preds_dict["delta_d_hat"])))
                    trainable_diff_sines = []
                    for i, parameter in enumerate(self.trainable_params):
                        param_int = int(parameter[6:])
                        trainable_diff_sines.append(param_diff_sines[:, param_int])
                        print("Parameter: " + str(parameter))
                        print("GT SMPL: " + str(data[1][:, param_int]))
                        print("Parameters: " + str(preds_dict["learned_params"][:, param_int]))
                        print("Parameter ang. MAE: " + str(param_diff_sines[:, param_int]))
                        if "delta_d_hat_mu" in preds_dict.keys():
                            print("Delta_d_hat_mu: " + str(preds_dict["delta_d_hat_mu"][:, param_int]))   # ProbCNN architecture only
                            print("Delta_d_hat_sigma: " + str(preds_dict["delta_d_hat_sigma"][:, param_int]))   # ProbCNN architecture only
                        print("Delta_d: " + str(preds_dict["delta_d"][:, param_int]))
                        print("Delta_d_hat: " + str(preds_dict["delta_d_hat"][:, param_int]))
                        #print("Delta_d_hat: " + str(preds_dict["delta_d_hat"][:, i+1]))
                        print("Difference sine: " + str(delta_d_diff_sines[:, param_int]))
                        #print("Difference sine: " + str(delta_d_diff_sines[:, i]))
                        #print("Delta_d_hat loss: " + str(preds_dict["delta_d_hat_mse"]))
                        #print("Difference sine (direct): " + str(np.sin(preds_dict["delta_d"] - preds_dict["delta_d_hat"])[:, param_int]))
                        #print("Difference sine (from normals): " + str(preds_dict["diff_angles"]))
                        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

                        #self.delta_d_log.write('parameter: ' + str(parameter) + '\n' + 'Delta_d: ' + str(preds[6][:, param_int]) + '\n')
                        self.delta_d_log.write('parameter: ' + str(parameter) + '\n' + 'Delta_d: ' + str(preds[7][:, param_int]) + '\n')

                    if "params_mse" in preds_dict.keys():
                        params_mse = np.mean(preds_dict["params_mse"], axis=0)
                        print("Params MSE: " + str(params_mse))
                        self.params_mse_log.write('epoch {:05d}\n'.format(epoch + 1))
                        self.params_mse_log.write(str(params_mse) + "\n")

                    #print("Predictions for first example: " + str(preds_dict["delta_d_hat"][0]))
                    if "rot3d_pose" in preds_dict.keys():
                        #print("rot3d_pose trace: " + str(np.trace(preds_dict["rot3d_pose"][0], axis1=1, axis2=2)))
                        #print("rot3d_delta_d_pose trace: " + str(np.trace(preds_dict["rot3d_delta_d_pose"][0], axis1=1, axis2=2)))
                        #print("rot3d_pose: " + str(preds_dict["rot3d_pose"][0]))   # RotConv1d architecture only
                        #print("rot3d_delta_d_pose: " + str(preds_dict["rot3d_delta_d_pose"][0]))   # RotConv1d architecture only
                        print("rot3d_pose error: " + str(preds_dict["rot3d_pose"][0] - preds_dict["rot3d_delta_d_pose"][0]))   # RotConv1d architecture only
                        #pass
                    if "mapped_pose" in preds_dict.keys():
                        #print("mapped_pose: " + str(preds_dict["mapped_pose"][0]))   # RotConv1d architecture only
                        #print("mapped_delta_d_pose: " + str(preds_dict["mapped_delta_d_pose"][0]))   # RotConv1d architecture only
                        print("mapped_pose error: " + str(preds_dict["mapped_pose"][0] - preds_dict["mapped_delta_d_pose"][0]))   # RotConv1d architecture only
                        pass
                    if "rodrigues_delta_d_pose" in preds_dict.keys():
                        #print("rodrigues pose error: " + str(preds_dict["rodrigues_delta_d_pose"][0] - preds_dict["delta_d_pose_vec"][0]))
                        pass

                    avg_diff_sines = np.mean(trainable_diff_sines, axis=0)

                    if epoch == -1:
                        # print parameters to file
                        gt_example_parameters = data[1]
                        pred_example_parameters = preds_dict["learned_params"]
                        diff_example_parameters = gt_example_parameters - pred_example_parameters
                        param_save_dir = self.pred_path + "/example_parameters/"
                        os.system('mkdir ' + str(param_save_dir))

                        np.savetxt(param_save_dir + "gt_params.txt", gt_example_parameters)
                        np.savetxt(param_save_dir + "pred_params.txt", pred_example_parameters)
                        np.savetxt(param_save_dir + "diff.txt", diff_example_parameters)

                    # Track resets
                    BLOCK_SIZE = self.data_samples / self.RESET_PERIOD
                    #print("BLOCK_SIZE " + str(BLOCK_SIZE))
                    BLOCKS = self.examples // BLOCK_SIZE
                    #print("BLOCKS " + str(BLOCKS))
                    #if (epoch - 1) < 0 or self.testing:
                    if epoch < 0 or self.testing:
                        was_reset = [False, False, False, False, False]
                    else:
                        #INDEX = (epoch - 1) % self.RESET_PERIOD
                        INDEX = epoch % self.RESET_PERIOD
                        #print("INDEX " + str(INDEX))
                        was_reset = [entry == INDEX for entry in BLOCKS]
                    #print("was_reset " + str(was_reset))
                    #exit(1)

                    silh_comp_list = []
                    for i, learned_pc in enumerate(preds[2], 1):
                        # Store the learned mesh
                        print_mesh(os.path.join(self.pred_path, "{}_epoch.{:05d}.pred_pc_{:03d}.obj".format(data_type, epoch + 1, i)), learned_pc, self.smpl.faces)

                        pred_silhouette = Mesh(pointcloud=learned_pc).render_silhouette(show=False)
                        #cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:03d}.pred_silh_{:03d}.png".format(data_type, epoch + 1, i)), pred_silhouette)
                        gt_silhouette = Mesh(pointcloud=data_dict["gt_pc"][i-1]).render_silhouette(show=False)

                        if True:
                        #if self.gt_silhouettes[data_type] is not None:
                            # Store predicted silhouette and the difference between it and the GT silhouette
                            #gt_silhouette = (self.gt_silhouettes[data_type][i-1] * 255).astype("uint8")
                            #gt_silhouette = self.gt_silhouettes[data_type][i-1].astype("uint8")
                            #print("gt_silhouette shape: " + str(gt_silhouette.shape))
                            #gt_silhouette = gt_silhouette.reshape((gt_silhouette.shape[0], gt_silhouette.shape[1]))
                            #cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:03d}.gt_silh_{:03d}.png".format(data_type, epoch + 1, i)), gt_silhouette)

                            diff_silh = (gt_silhouette != pred_silhouette)*255
                            #diff_silh = abs(gt_silhouette - pred_silhouette)
                            #print(diff_silh.shape)
                            #cv2.imshow("Diff silh", diff_silh)
                            #cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:03d}.diff_silh_{:03d}.png".format(data_type, epoch + 1, i)), diff_silh.astype("uint8"))
                            silh_comp = np.concatenate([gt_silhouette, pred_silhouette, diff_silh], axis=1)
                            #cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:05d}.silh_comp_{:03d}.png".format(data_type, epoch + 1, i)), silh_comp.astype("uint8"))

                            if was_reset[i-1]:
                                # Grey the image
                                silh_comp /= 2

                            # Convert to rgb and write the difference sine to the image
                            silh_comp_rgb = np.zeros((silh_comp.shape[0], silh_comp.shape[1], 3))
                            for c in range(3):
                                silh_comp_rgb[:, :, c] = silh_comp

                            # Write to the image
                            font                   = cv2.FONT_HERSHEY_SIMPLEX
                            bottomLeftCorner       = (550,30)
                            gt_main_rot            = (0, 70)
                            pred_main_rot          = (0, 50)
                            delta_d_hat_pos        = (0, 90)
                            fontScale              = 0.6
                            fontColor              = (0,0,255)
                            lineType               = 2
                            cv2.putText(silh_comp_rgb, "Ang. MAE: {0:.3f}".format(avg_diff_sines[i-1]),
                                    bottomLeftCorner,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                            cv2.putText(silh_comp_rgb, "Main rot.: " +str(preds_dict["learned_params"][i-1, 0:3]),
                                    pred_main_rot,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                            cv2.putText(silh_comp_rgb, "GT Main rot.: " +str(data[1][i-1, 0:3]),
                                    gt_main_rot,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)

                            cv2.putText(silh_comp_rgb, "delta_d_hat: " +str(preds_dict["delta_d_hat"][i-1, 0:3]),
                                    delta_d_hat_pos,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                            # Add image to list
                            silh_comp_list.append(silh_comp_rgb)

                        # Save the predicted point cloud relative to the GT point cloud
                        print_mesh(os.path.join(self.pred_path, "{}_epoch.{:05d}.gt_pc_{:03d}.obj".format(data_type, epoch + 1, i)), data[2][i-1], self.smpl.faces)
                        print_point_clouds(os.path.join(self.pred_path, "{}_epoch.{:05d}.comparison_{:03d}.obj".format(data_type, epoch + 1, i)), [learned_pc, data[2][i-1]], [(255,0,0),(0,255,0)])

                    if len(silh_comp_list) > 0:
                        silh_comps_rgb = np.concatenate(silh_comp_list, axis=0)

                        font                   = cv2.FONT_HERSHEY_SIMPLEX
                        topLeftCorner          = (30,30)
                        fontScale              = 1
                        fontColor              = (0,0,255)
                        lineType               = 2

                        if self.testing:
                            text = "Iteration "
                        else:
                            text = "Epoch "

                        cv2.putText(silh_comps_rgb, text + str(epoch + 1),
                                    topLeftCorner,
                                    font,
                                    fontScale,
                                    fontColor,
                                    lineType)
                        cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:05d}.silh_comps.png".format(data_type, epoch + 1)), silh_comps_rgb.astype("uint8"))


class OptLearnerLossGraphCallback(tf.keras.callbacks.Callback):
    def __init__(self, run_dir, graphing_period=100):
        self.run_dir = run_dir
        self.graphing_period = graphing_period
        os.system("mkdir " + str(self.run_dir) + "plots/")

    def plot_losses(self, epoch, show=False):
        loss_path = self.run_dir + "/logs/losses.txt"

        losses_to_load = ["loss", "delta_d_hat_mse_loss", "pc_mean_euc_dist_loss", "delta_d_mse_loss", "diff_angle_mse_loss"]
        loss_alias = ["loss", "deep opt. loss", "point cloud loss", "smpl loss", "angle loss"]
        loss_display_name = [alias + " (" + losses_to_load[i] + ")" for i, alias in enumerate(loss_alias)]
        loss_display_name_dict = {losses_to_load[i]: loss_display_name[i] for i in range(len(losses_to_load))}
        scale_presets = [0.001, 1, 1, 1, 1]
        scale_presets_dict = {losses_to_load[i]: scale_presets[i] for i in range(len(losses_to_load))}
        #style_presets = [".", "+", "v", "s", ","]
        style_preset = '.'
        color_presets = ["r", "b", "g", "k", "y"]
        color_presets_dict = {losses_to_load[i]: color_presets[i] for i in range(len(losses_to_load))}

        column_names = []
        styles = []
        colors = []
        if epoch <= 100:
            SUBSAMPLE_PERIOD = 10
        else:
            SUBSAMPLE_PERIOD = 20

        # Load the loss files
        dir_losses = {}
        for loss in losses_to_load:
            dir_losses[loss] = []
        with open(loss_path, mode='r') as f:
            for i, s in enumerate(f.readlines()):
                if i > 0:
                    s = json.loads(s)
                    if s["epoch"] % SUBSAMPLE_PERIOD == 0:
                        for loss in losses_to_load:
                            dir_losses[loss].append(float(s.get(loss)))

        for loss_name, loss_values in dir_losses.items():
            plt.plot(np.arange(len(loss_values))*SUBSAMPLE_PERIOD, scale_presets_dict[loss_name]*np.array(loss_values), markersize=6, linewidth=1, marker=style_preset, color=color_presets_dict[loss_name], label=loss_display_name_dict[loss_name])
        plt.ylabel("Loss value", fontsize=16)
        plt.xlabel("Epoch", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(prop={'size': 8})
        plt.savefig("{}/plots/plot_E{:05d}.png".format(self.run_dir, epoch))
        if show:
            plt.show()
        plt.clf()

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and epoch % self.graphing_period == 0:
            try:
                self.plot_losses(epoch)
            except Exception as exc:
                print("Plotting failed with exception: '{}'".format(exc))


class OptimisationCallback(tf.keras.callbacks.Callback):
    def __init__(self, opt_period, opt_epochs, lr, log_path, smpl_model, training_data, train_inputs=[[None], [None], [None]], train_silh=None, val_inputs=[[None], [None], [None]], val_silh=None, test_inputs=[[None], [None], [None]], test_silh=None, pred_path="../", period=1, trainable_params=[], visualise=True):
        self.pred_callback = OptLearnerPredOnEpochEnd(log_path, smpl_model, train_inputs, train_silh, val_inputs, val_silh, test_inputs, test_silh, pred_path, period, trainable_params, visualise)

        self.training_data = training_data
        self.pred_path = pred_path
        self.log_path = log_path
        self.opt_period = opt_period
        self.lr = lr
        self.opt_epochs = opt_epochs
        self.trainable_params = trainable_params

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        """ Perform the optimisation """
        epoch = int(epoch)
        if (epoch + 1) % self.opt_period == 0:
            # Prepare save paths
            epoch_pred_path = self.pred_path + "epoch_{:02d}/".format(epoch + 1)
            epoch_log_path = self.log_path + "epoch_{:02d}/".format(epoch)
            os.mkdir(epoch_pred_path)
            os.mkdir(epoch_log_path)
            print("\nStarting optimisation...")
            print("Saving to directory {}".format(epoch_pred_path))

            # Optimise
            self.pred_callback.pred_path = epoch_pred_path
            self.pred_callback.log_path = epoch_log_path
            self.optimise()

    def optimise(self):
        assert False, "Should not be called while training"
        print("Optimising for {} epochs with lr {}...".format(self.opt_epochs, self.lr))
        optlearner_model = copy(self.model)
        metrics_names = optlearner_model.metrics_names
        #print("metrics_names: " + str(metrics_names))
        #named_scores = {}

        param_ids = ["param_{:02d}".format(i) for i in range(85)]
        param_trainable = { param: (param in self.trainable_params) for param in param_ids }
        trainable_layers_names = [param_layer for param_layer, trainable in param_trainable.items() if trainable]
        trainable_layers = {optlearner_model.get_layer(layer_name): int(layer_name[6:8]) for layer_name in trainable_layers_names}

        self.pred_callback.set_model(optlearner_model)
        for epoch in range(self.opt_epochs):
            print("Iteration: " + str(epoch + 1))
            print("----------------------")
            self.pred_callback.on_epoch_begin(epoch=int(epoch), logs=None)
            #print('emb layer weights'+str(emb_weights))
	    #print('shape '+str(emb_weights[0].shape))
	    y_pred = optlearner_model.predict(self.training_data)
            #smpl_update = y_pred[7]
            smpl_update = y_pred[8]

            for emb_layer, param_num in trainable_layers.items():
	        emb_weights = emb_layer.get_weights()
                emb_weights += self.lr * np.array(smpl_update[:, param_num]).reshape((smpl_update.shape[0], 1))
	        emb_layer.set_weights(emb_weights)

            # Evaluate model performance
            #scores = optlearner_model.evaluate(x_test, y_test, batch_size=100)
            #for i, score in enumerate(scores):
            #    named_scores[metrics_names[i]] = score
            #print("scores: " + str(scores))
            #exit(1)
            self.pred_callback.set_model(optlearner_model)
            self.pred_callback.on_epoch_end(epoch=int(epoch), logs=None)

        print("Optimisation complete.")


class PredOnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, log_path, smpl_model, x_train=None, y_train=None, x_val=None, y_val=None, x_test=None, y_test=None, pred_path="../", period=5, visualise=True):
        # Open the log file
        log_path = os.path.join(log_path, "losses.txt")
        self.epoch_log = open(log_path, mode='wt', buffering=1)

        # Model to use to create meshes from SMPL parameters
        self.smpl = smpl_model

        # Store path for prediction visualisations
        self.pred_path = pred_path

        # Store data to be used for training examples
        self.pred_data = {"train": x_train, "val": x_val, "test": x_test}
        self.gt_pc = {"train": y_train, "val": y_val, "test": y_test}

        # Store the prediction period
        self.period = period

        # Store whether to visualise data during training
        self.visualise = visualise

        # Store model
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        """ Store the model loss and accuracy at the end of every epoch, and store a model prediction on data """
        self.epoch_log.write(json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n')

        if (epoch + 1) % self.period == 0 or epoch == 0:
            # Predict on all of the given silhouettes
            for data_type, data in self.pred_data.items():
                if data is not None:
                    if not isinstance(data, list) or type(data) == np.array:
                        data = np.array(data)
                        data = data.reshape((1, data.shape[0], data.shape[1], data.shape[2]))

                    #for i, silhouette in enumerate(data):
                    #    # Save silhouettes
                    #    silhouette *= 255
                    #    cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:03d}.gt_silh_{:03d}.png".format(data_type, epoch + 1, i)), silhouette.astype("uint8"))

                    preds = self.model.predict(data)
                    #print("Predictions: " + str(preds))

                    for i, pred in enumerate(preds[1], 1):
                        #self.smpl.set_params(pred[:72].reshape((24, 3)), pred[72:82], pred[82:])
                        #self.smpl.save_to_obj(os.path.join(self.pred_path, "{}_pred_{:03d}.obj".format(data_type, i)))
                        #print_mesh(os.path.join(self.pred_path, "epoch.{:03d}.{}_gt_{:03d}.obj".format(epoch, data_type, i)), gt[i-1], smpl.faces)
                        print_mesh(os.path.join(self.pred_path, "{}_epoch.{:03d}.pred_{:03d}.obj".format(data_type, epoch + 1, i)), pred, self.smpl.faces)

                        # Store predicted silhouette and the difference between it and the GT silhouette
                        gt_silhouette = (data[i-1] * 255).astype("uint8").reshape(data.shape[1], data.shape[2])
                        #cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:03d}.gt_silh_{:03d}.png".format(data_type, epoch + 1, i)), gt_silhouette)

                        pred_silhouette = Mesh(pointcloud=pred).render_silhouette(show=False)
                        #cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:03d}.pred_silh_{:03d}.png".format(data_type, epoch + 1, i)), pred_silhouette)

                        diff_silh = abs(gt_silhouette - pred_silhouette)
                        #print(diff_silh.shape)
                        #cv2.imshow("Diff silh", diff_silh)
                        #cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:03d}.diff_silh_{:03d}.png".format(data_type, epoch + 1, i)), diff_silh.astype("uint8"))
                        silh_comp = np.concatenate([gt_silhouette, pred_silhouette, diff_silh])
                        cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:03d}.silh_comp_{:03d}.png".format(data_type, epoch + 1, i)), silh_comp.astype("uint8"))

                        if self.gt_pc[data_type] is not None:
                            print_mesh(os.path.join(self.pred_path, "{}_epoch.{:03d}.gt_pc_{:03d}.obj".format(data_type, epoch + 1, i)),self.gt_pc[data_type], self.smpl.faces)
                            print_point_clouds(os.path.join(self.pred_path, "{}_epoch.{:03d}.comparison_{:03d}.obj".format(data_type, epoch + 1, i)), [pred, self.gt_pc[data_type]], [(255,0,0),(0,255,0)])


                    if self.visualise:
                        # Show a random sample
                        rand_index = np.random.randint(low=0, high=len(data)) + 1
                        mesh = Mesh(filepath=os.path.join(self.pred_path, "{}_epoch.{:03d}.pred_{:03d}.obj".format(data_type, epoch + 1, rand_index)))

                        # Show the true silhouette
                        true_silh = data[rand_index - 1]
                        true_silh = true_silh.reshape(true_silh.shape[:-1])
                        plt.imshow(true_silh, cmap='gray')
                        plt.title("True {} silhouette {:03d}".format(data_type, rand_index))
                        plt.show()

                        # Show the predicted silhouette and mesh
                        mesh.render_silhouette(title="Predicted {} silhouette {:03d}".format(data_type, rand_index))
                        diff_silh = cv2.imread("{}_epoch.{:03d}.diff_silh_{:03d}.png".format(data_type, epoch + 1, rand_index))
                        cv2.imshow("Predicted {} silhouette {:03d}".format(data_type, rand_index), diff_silh)

                        try:
                            mesh.render3D()
                        except Exception:
                            pass
