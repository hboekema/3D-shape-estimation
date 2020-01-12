import os
import sys
import json
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from render_mesh import Mesh
from smpl_np_rot_v6 import print_mesh, print_point_clouds



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
