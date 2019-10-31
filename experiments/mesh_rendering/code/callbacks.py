import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from render_mesh import Mesh

#class PredOnEpochEnd(tf.keras.callbacks.Callback):
class PredOnEpochEnd():
    def __init__(self, log_path, smpl_model, x_train=None, x_val=None, x_test=None, pred_path="../", period=5, run_id="no id", visualise=True):
        # Open the log file
        self.run_id = run_id
        log_path = os.path.join(log_path, "losses[{}].txt".format(self.run_id))
        self.epoch_log = open(log_path, mode='wt', buffering=1)

        # Model to use to create meshes from SMPL parameters
        self.smpl = smpl_model

        # Store path for prediction visualisations
        self.pred_path = pred_path

        # Store data to be used for training examples
        self.pred_data = {"train": x_train, "val": x_val, "test": x_test}

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

        if epoch % self.period == 0:
            # Predict on all of the given silhouettes
            for data_type, data in self.pred_data.items():
                if data is not None:
                    if not isinstance(data, list) or type(data) == np.array:
                        data = np.array(data)
                        data = data.reshape((1, *data.shape))

                    preds = np.array(self.model.predict_(data))
                    for i, pred in enumerate(preds, 1):
                        self.smpl.set_params(pred[:72].reshape((24, 3)), pred[72:82], pred[82:])
                        self.smpl.save_to_obj(os.path.join(self.pred_path,
                                                           "{}_pred_{:03d}[{}].obj".format(data_type, i, self.run_id)))

                    if self.visualise:
                        # Show a random sample
                        rand_index = np.random.randint(low=0, high=len(data)) + 1
                        mesh = Mesh(filepath=os.path.join(self.pred_path,
                                                   "{}_pred_{:03d}[{}].obj".format(data_type, rand_index, self.run_id)))

                        # Show the true silhouette
                        true_silh = data[rand_index - 1]
                        true_silh = true_silh.reshape(true_silh.shape[:-1])
                        plt.imshow(true_silh, cmap='gray')
                        plt.title("True {} silhouette {:03d}".format(data_type, rand_index))
                        plt.show()

                        # Show the predicted silhouette and mesh
                        mesh.render_silhouette(title="Predicted {} silhouette {:03d}".format(data_type, rand_index))
                        try:
                            mesh.render3D()
                        except Exception:
                            pass
