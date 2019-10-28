import os
import json
import numpy as np
import tensorflow as tf


class PredOnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, log_path, smpl_model, x_train=None, x_val=None, x_test=None, pred_path="../", run_id="no id"):
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

    def on_epoch_end(self, epoch, logs=None):
        """ Store the model loss and accuracy at the end of every epoch, and store a model prediction on data """
        self.epoch_log.write(json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n')

        # Predict on all of the given silhouettes
        for data_type, data in self.pred_data.items():
            if data is not None:
                if not isinstance(data, list) or type(data) == np.array:
                    data = np.array(data)
                    data = data.reshape((1, *data.shape))

                preds = self.model.predict(data)
                for i, pred in enumerate(preds, 1):
                    self.smpl.set_params(pred[:72].reshape((24, 3)), pred[72:82], pred[82:])
                    self.smpl.save_to_obj(os.path.join(self.pred_path,
                                                       "{}_pred_{:03d}[{}].obj".format(data_type, i, self.run_id)))

