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
import pandas as pd
from datetime import datetime

sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from render_mesh import Mesh
from smpl_np_rot_v6 import print_mesh, print_point_clouds
from tools.rotation_helpers import geodesic_error
from tools.log_util import LogFile


class OptLearnerPredOnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, log_path, smpl_model, train_inputs=[[None], [None], [None]], train_silh=None, val_inputs=[[None], [None], [None]], val_silh=None, test_inputs=[[None], [None], [None]], test_silh=None, pred_path="../", period=5, trainable_params=[], visualise=True, testing=False, RESET_PERIOD=10, data_samples=10000, train_gen=None, val_gen=None, test_gen=None, ARCHITECTURE=None, losses=[], loss_weights=[], generator=None):
        # Open the log files
        #epoch_log_path = os.path.join(log_path, "losses.txt")
        #self.epoch_log = open(epoch_log_path, mode='wt', buffering=1)
        self.epoch_log = LogFile()._create_log(log_path, "losses.txt")

        # Model to use to create meshes from SMPL parameters
        self.smpl = smpl_model

        # Store path for prediction visualisations
        self.pred_path = pred_path

        # Store data to be used for examples
        self.input_data = {"train": train_inputs, "val": val_inputs, "test": test_inputs}
        self.gt_silhouettes = {"train": train_silh, "val": val_silh, "test": test_silh}
        self.generator_paths = {"train": train_gen, "val": val_gen, "test": test_gen}
        self.generator = generator

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

        # Architecture
        self.ARCHITECTURE = ARCHITECTURE
        self.losses = losses
        self.loss_weights = loss_weights

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
                    print("Current time: " + str(datetime.now()))
                    print("Active losses: " + str(self.losses))
                    print("Active loss weights: " + str(self.loss_weights))
                    # Predict on these input parameters
                    #print("data value: " + str(data))
                    gen_path = self.generator_paths[data_type]
                    additional_input = None
                    if gen_path is not None:
                        gen_path = gen_path + "cb_samples_E{}.npz".format(epoch)
                        try:
                        #if True:
                            print("Starting loading npz at " + str(datetime.now()))
                            #assert os.path.isfile(gen_path), "file doesn't exist: " + str(gen_path)
                            with np.load(gen_path, allow_pickle=True) as temp_data:
                                #print(temp_data.keys())
                                if "trainable_params" in temp_data.keys():
                                    data = [temp_data["indices"], temp_data["params"], temp_data["pcs"], temp_data["trainable_params"]]
                                    additional_input = "trainable_params"
                                elif "params_to_train" in temp_data.keys():
                                    data = [temp_data["indices"], temp_data["params"], temp_data["pcs"], temp_data["params_to_train"]]
                                    additional_input = "params_to_train"
                                else:
                                    data = [temp_data["indices"], temp_data["params"], temp_data["pcs"]]
                            print("Finished loading npz at " + str(datetime.now()))
                        except Exception as e:
                            print("Skipping - load failed with exception '{}'".format(e))
                        #    #exit(1)
                            return None
                    #if self.generator is not None:
                    #    data, _ = self.generator.yield_data(epoch)

                    #print("Rendering in callback...")
                    X_silh = []
                    for pc in data[2]:
                        # Render the silhouette from the point cloud
                        silh = Mesh(pointcloud=pc).render_silhouette(dim=[128, 128], show=False)
                        X_silh.append(silh)
                    #print("Finished rendering.")
                    #input("Waiting...")

                    data_dict = {"embedding_index": np.array(data[0]), "gt_params": np.array(data[1]), "gt_pc": np.array(data[2]), "gt_silh": np.array(X_silh)}
                    #print(data_dict)
                    if self.ARCHITECTURE == "PeriodicOptLearnerArchitecture":
                        additional_input = "params_to_train"
                    if self.ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture":
                        additional_input = "trainable_params"
                    if additional_input is not None:
                        data_dict[additional_input] = np.array(data[4])
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

                    param_diff_sines = np.abs(np.sin(0.5*(data[1] - preds_dict["learned_params"])))
                    delta_d_diff_sines = np.abs(np.sin(0.5*(preds_dict["delta_d"] - preds_dict["delta_d_hat"])))
                    trainable_diff_sines = []
                    for i, parameter in enumerate(self.trainable_params):
                        param_int = int(parameter[6:8])
                        trainable_diff_sines.append(param_diff_sines[:, param_int])
                        #if False:
                        if True:
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

                    #print("Predictions for first example: " + str(preds_dict["delta_d_hat"][0]))
                    if "rot3d_pose" in preds_dict.keys():
                        #print("rot3d_pose trace: " + str(np.trace(preds_dict["rot3d_pose"][0], axis1=1, axis2=2)))
                        #print("rot3d_delta_d_pose trace: " + str(np.trace(preds_dict["rot3d_delta_d_pose"][0], axis1=1, axis2=2)))
                        #print("rot3d_pose: " + str(preds_dict["rot3d_pose"][0]))   # RotConv1d architecture only
                        #print("rot3d_delta_d_pose: " + str(preds_dict["rot3d_delta_d_pose"][0]))   # RotConv1d architecture only
                        #print("rot3d_pose error: " + str(preds_dict["rot3d_pose"][0] - preds_dict["rot3d_delta_d_pose"][0]))   # RotConv1d architecture only
                        pass
                    if "mapped_pose" in preds_dict.keys():
                        #print("mapped_pose: " + str(preds_dict["mapped_pose"][0]))   # RotConv1d architecture only
                        #print("mapped_delta_d_pose: " + str(preds_dict["mapped_delta_d_pose"][0]))   # RotConv1d architecture only
                        #print("mapped_pose error: " + str(preds_dict["mapped_pose"][0] - preds_dict["mapped_delta_d_pose"][0]))   # RotConv1d architecture only
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

                    if self.ARCHITECTURE == "ShapeConv2DOptLearnerArchitecture":
                        np.save(self.pred_path + "{}_epoch.{:05d}.gt_normals_pred_{:03d}.npy".format(data_type, epoch + 1, i), preds_dict["gt_normals_pred"])
                        np.save(self.pred_path + "{}_epoch.{:05d}.gt_normals_TRUE_shape_{:03d}.npy".format(data_type, epoch + 1, i), preds_dict["gt_cross_product"])
                        np.save(self.pred_path + "{}_epoch.{:05d}.gt_normals_TRUE_no_shape_{:03d}.npy".format(data_type, epoch + 1, i), preds_dict["gt_no_shape_cross_product"])

                    # Track resets
                    BLOCK_SIZE = self.data_samples / self.RESET_PERIOD
                    #print("BLOCK_SIZE " + str(BLOCK_SIZE))
                    BLOCKS = self.examples // BLOCK_SIZE
                    #print("BLOCKS " + str(BLOCKS))
                    #if (epoch - 1) < 0 or self.testing:
                    if epoch < 0 or self.testing:
                        was_reset = [False for _ in BLOCKS]
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

                        pred_silhouette = Mesh(pointcloud=learned_pc).render_silhouette(dim=[256, 256], show=False)
                        #cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:03d}.pred_silh_{:03d}.png".format(data_type, epoch + 1, i)), pred_silhouette)
                        gt_silhouette = Mesh(pointcloud=data_dict["gt_pc"][i-1]).render_silhouette(dim=[256, 256], show=False)

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
                        ang_mae                = (550,30)
                        normals_loss           = (550,240)
                        gt_main_rot            = (0, 70)
                        pred_main_rot          = (0, 50)
                        delta_d_hat_pos        = (0, 90)
                        fontScale              = 0.6
                        fontColor              = (0,0,255)
                        lineType               = 2
                        #cv2.putText(silh_comp_rgb, "Ang. MAE: {0:.3f}".format(avg_diff_sines[i-1]),
                        #        ang_mae,
                        #        font,
                        #        fontScale,
                        #        fontColor,
                        #        lineType)

                        #cv2.putText(silh_comp_rgb, "Norm. Loss: {0:.3f}".format(np.mean(preds_dict["diff_angle_mse"][i-1])),
                        #        normals_loss,
                        #        font,
                        #        fontScale,
                        #        fontColor,
                        #        lineType)

                        #cv2.putText(silh_comp_rgb, "Main rot.: " +str(preds_dict["learned_params"][i-1, 0:3]),
                        #        pred_main_rot,
                        #        font,
                        #        fontScale,
                        #        fontColor,
                        #        lineType)

                        #cv2.putText(silh_comp_rgb, "GT Main rot.: " +str(data[1][i-1, 0:3]),
                        #        gt_main_rot,
                        #        font,
                        #        fontScale,
                        #        fontColor,
                        #        lineType)

                        #cv2.putText(silh_comp_rgb, "delta_d_hat: " +str(preds_dict["delta_d_hat"][i-1, 0:3]),
                        #        delta_d_hat_pos,
                        #        font,
                        #        fontScale,
                        #        fontColor,
                        #        lineType)
                        # Add image to list
                        silh_comp_list.append(silh_comp_rgb)

                        # Save the predicted point cloud relative to the GT point cloud
                        print_mesh(os.path.join(self.pred_path, "{}_epoch.{:05d}.gt_pc_{:03d}.obj".format(data_type, epoch + 1, i)), data[2][i-1], self.smpl.faces)
                        if self.ARCHITECTURE == "ShapeConv2DOptLearnerArchitecture":
                            print_mesh(os.path.join(self.pred_path, "{}_epoch.{:05d}.pred_pc_no_shape_{:03d}.obj".format(data_type, epoch + 1, i)), preds_dict["optlearner_pc_no_shape"][i-1], self.smpl.faces)
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

                        #cv2.putText(silh_comps_rgb, text + str(epoch + 1),
                        #            topLeftCorner,
                        #            font,
                        #            fontScale,
                        #            fontColor,
                        #            lineType)
                        cv2.imwrite(os.path.join(self.pred_path, "{}_epoch.{:05d}.silh_comps.png".format(data_type, epoch + 1)), silh_comps_rgb.astype("uint8"))



class GeneratorParamErrorCallback(tf.keras.callbacks.Callback):
    def __init__(self, run_dir, generator, period=10, ARCHITECTURE=None, num_samples=5):
        self.log_path = run_dir + "logs/"

        self.params_mse_log = LogFile()._create_log(self.log_path, "params_mse.txt")
        self.params_mspe_log = LogFile()._create_log(self.log_path, "params_mspe.txt")
        self.params_angle_log = LogFile()._create_log(self.log_path, "params_angle.txt")

        self.gt_normals_log = LogFile()._create_log(self.log_path, "gt_normals.txt")
        self.opt_normals_log = LogFile()._create_log(self.log_path, "opt_normals.txt")
        self.mse_normals_log = LogFile()._create_log(self.log_path, "mse_normals.txt")
        self.network_mse_normals_log = LogFile()._create_log(self.log_path, "network_mse_normals.txt")

        self.delta_d_hat_log = LogFile()._create_log(self.log_path, "delta_d_hat.txt")
        self.delta_angle_log = LogFile()._create_log(self.log_path, "delta_angle.txt")
        self.delta_d_log = LogFile()._create_log(self.log_path, "delta_d.txt")
        self.delta_d_abs_log = LogFile()._create_log(self.log_path, "delta_d_magnitude.txt")

        self.gt_normals_LOSS_log = LogFile()._create_log(self.log_path, "gt_normals_LOSS.txt")

        self.generator = generator
        assert period > 0
        self.period = period
        self.num_samples = num_samples
        self.epsilon = 1e-3

        self.ARCHITECTURE = ARCHITECTURE

    #def _create_log(self, name):
    #    log_full_path = os.path.join(self.log_path, name)
    #    os.system("touch " + log_full_path)
    #    log = open(log_full_path, mode='wt', buffering=1)
    #    return log

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period == 0:
            data, _ = self.generator.yield_data(epoch)

            data_dict = {"embedding_index": np.array(data[0]), "gt_params": np.array(data[1]), "gt_pc": np.array(data[2]), "gt_silh": np.array(data[3])}
            if self.ARCHITECTURE == "PeriodicOptLearnerArchitecture":
                data_dict["params_to_train"] = np.array(data[3])
            if self.ARCHITECTURE == "NewDeepConv1DOptLearnerArchitecture":
                data_dict["trainable_params"] = np.array(data[3])
            preds = self.model.predict(data_dict)

            # Process outputs to be easy to read
            metrics_names = self.model.metrics_names[:-1]
            output_names = [metric[:-5] for i, metric in enumerate(metrics_names) if i > 0]
            #print(output_names)
            preds_dict = {output_name: preds[i] for i, output_name in enumerate(output_names)}

            # Calculate values to store
            delta_d_hat = preds_dict["delta_d_hat"]
            #delta_d_hat_mean = np.mean(delta_d_hat, axis=0)
            delta_d_hat_samples = delta_d_hat[:self.num_samples]
            self.delta_d_hat_log.write('epoch {:05d}\n'.format(epoch + 1))
            self.delta_d_hat_log.write(str(delta_d_hat_samples) + "\n")

            gt_params = data_dict["gt_params"]
            optlearner_params = preds_dict["learned_params"]
            delta_angle = geodesic_error(gt_params, optlearner_params)
            delta_angle = np.mean(delta_angle, axis=0)
            self.delta_angle_log.write('epoch {:05d}\n'.format(epoch + 1))
            self.delta_angle_log.write(str(delta_angle) + "\n")

            delta_d = preds_dict["delta_d"]
            #delta_d_mean = np.mean(delta_d, axis=0)
            delta_d_samples = delta_d[:self.num_samples]
            self.delta_d_log.write('epoch {:05d}\n'.format(epoch + 1))
            self.delta_d_log.write(str(delta_d_samples) + "\n")

            delta_d_abs = np.mean(abs(preds_dict["delta_d"]), axis=0)
            #print("Delta_d_abs: " + str(delta_d_abs))
            self.delta_d_abs_log.write('epoch {:05d}\n'.format(epoch + 1))
            self.delta_d_abs_log.write(str(delta_d_abs) + "\n")

            params_angle = geodesic_error(delta_d, delta_d_hat)
            params_angle = np.mean(params_angle, axis=0)
            self.params_angle_log.write('epoch {:05d}\n'.format(epoch + 1))
            self.params_angle_log.write(str(params_angle) + "\n")

            if "params_mse" in preds_dict.keys():
                params_mse = np.mean(preds_dict["params_mse"], axis=0)
                #print("Params MSE: " + str(params_mse))
                self.params_mse_log.write('epoch {:05d}\n'.format(epoch + 1))
                self.params_mse_log.write(str(params_mse) + "\n")

                params_mspe = np.mean(preds_dict["params_mse"]/(delta_d_abs + self.epsilon), axis=0)
                #print("Params MSPE: " + str(params_mspe))
                self.params_mspe_log.write('epoch {:05d}\n'.format(epoch + 1))
                self.params_mspe_log.write(str(params_mspe) + "\n")

            if "gt_cross_product" in preds_dict.keys():
                gt_normals = preds_dict["gt_cross_product"][:self.num_samples]
                #print("GT normals: " + str(gt_normals))
                self.gt_normals_log.write('epoch {:05d}\n'.format(epoch + 1))
                self.gt_normals_log.write(str(gt_normals) + "\n")

                opt_normals = preds_dict["opt_cross_product"][:self.num_samples]
                #print("Opt. normals: " + str(opt_normals))
                self.opt_normals_log.write('epoch {:05d}\n'.format(epoch + 1))
                self.opt_normals_log.write(str(opt_normals) + "\n")

                mse_normals = np.mean(np.square(gt_normals - opt_normals), axis=-1)[:self.num_samples]
                #print("mse. normals: " + str(mse_normals))
                self.mse_normals_log.write('epoch {:05d}\n'.format(epoch + 1))
                self.mse_normals_log.write(str(mse_normals) + "\n")

                network_mse_normals = preds_dict["diff_angle_mse"][:self.num_samples]
                #print("Cross normals MSE: " + str(network_mse_normals))
                self.network_mse_normals_log.write('epoch {:05d}\n'.format(epoch + 1))
                self.network_mse_normals_log.write(str(network_mse_normals) + "\n")

            if "gt_normals_LOSS" in preds_dict.keys():
                gt_normals_LOSS = np.mean(preds_dict["gt_normals_LOSS"])
                self.gt_normals_LOSS_log.write('epoch {:05d}\n'.format(epoch + 1))
                self.gt_normals_LOSS_log.write(str(gt_normals_LOSS) + "\n")


class TestingCallback:
    def __init__(self, dir_path, input_names=None, output_names=None):
        self.dir_path =  dir_path

        self.input_names = input_names
        self.output_names = output_names

    def set_names(self, input_names=None, output_names=None):
        if input_names is not None:
            self.input_names = input_names
        if output_names is not None:
            self.output_names = output_names

    def store_results(self, epoch, data, preds):
        data_dict = {self.input_names[i]: value for i, value in enumerate(data)}
        preds_dict = {self.output_names[i]: value for i, value in enumerate(preds)}

        # Calculate values to store
        gt_params = data_dict["gt_params"]
        optlearner_params = preds_dict["learned_params"]
        delta_d_hat = preds_dict["delta_d_hat"]
        #delta_angle_all = geodesic_error(gt_params, optlearner_params)
        #delta_angle = np.mean(delta_angle_all, axis=0)
        #delta_d = preds_dict["delta_d"]

        save_path = self.dir_path + "data_E{:03d}.npz".format(epoch)
        #np.savez(save_path, gt_params=gt_params, opt_params=optlearner_params, delta_d=delta_d, delta_d_hat=delta_d_hat)
        np.savez(save_path, gt_params=gt_params, opt_params=optlearner_params, delta_d_hat=delta_d_hat)


class TestingLogCallback:
    def __init__(self, log_dir, input_names=None, output_names=None):
        self.log_path =  log_dir

        self.gt_params_log = self._create_log("gt_params.csv")
        self.opt_params_log = self._create_log("opt_params.csv")
        self.delta_d_log = self._create_log("delta_d.csv")
        self.delta_d_hat_log = self._create_log("delta_d_hat.csv")
        self.delta_angle_log = self._create_log("delta_angle.csv")

        self.input_names = input_names
        self.output_names = output_names

    def _create_log(self, name):
        log_full_path = os.path.join(self.log_path, name)
        os.system("touch " + log_full_path)
        log = open(log_full_path, mode='a', buffering=1)
        return log

    def _write_to_csv(self, log, values):
        panel = pd.Panel(values).to_frame().stack().reset_index()
        panel.columns = ["iteration", "sample", "parameter", "value"]

        panel.to_csv(log, index=False)

    def set_names(self, input_names=None, output_names=None):
        if input_names is not None:
            self.input_names = input_names
        if output_names is not None:
            self.output_names = output_names

    def store_results(self, epoch, data, preds):
        data_dict = {self.input_names[i]: value for i, value in enumerate(data)}
        preds_dict = {self.output_names[i]: value for i, value in enumerate(preds)}

        # Calculate values to store
        gt_params = data_dict["gt_params"]
        self.gt_params_log.write(gt_params)

        optlearner_params = preds_dict["learned_params"]
        self.opt_params_log.write(optlearner_params)

        delta_d_hat = preds_dict["delta_d_hat"]
        self.delta_d_hat_log.write(delta_d_hat)

        delta_angle = geodesic_error(gt_params, optlearner_params)
        delta_angle = np.mean(delta_angle, axis=0)
        self.delta_angle_log.write(delta_angle)

        delta_d = preds_dict["delta_d"]
        self.delta_d_log.write(delta_d)


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
