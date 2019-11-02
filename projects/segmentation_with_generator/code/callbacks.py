""" Model callback functions """

import os
import tensorflow as tf
import numpy as np
import json
import cv2


class PredOnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, log_path, x_train=None, x_val=None, x_test=None, pred_path="../", run_id="no id"):
        # Open the log file
        self.run_id = run_id
        log_path = os.path.join(log_path, "losses[{}].txt".format(self.run_id))
        self.epoch_log = open(log_path, mode='wt', buffering=1)

        # Store path for prediction visualisations
        self.pred_path = pred_path

        # Store data to be used for training examples
        self.pred_data = {"train": x_train, "val": x_val, "test": x_test}

    def on_epoch_end(self, epoch, logs=None):
        """ Store the model loss and IOU score at the end of every epoch, and store a model prediction on data """
        self.epoch_log.write(json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n')

        # Predict on all of the given images
        for data_type, data in self.pred_data.items():
            if data is not None:
                if isinstance(data, list) or type(data) == np.array:
                    # @TODO: generalise this to any W, H and n_channels
                    preds = self.model.predict(data).reshape((data.shape[0], 256, 256))
                    preds *= 255
                    preds = preds.astype(np.uint8)
                    for i, pred in enumerate(preds, 1):
                        cv2.imwrite(os.path.join(self.pred_path, "{}-img{}.{}[{}].png"
                                                 .format(data_type, i, epoch+1, self.run_id)), pred*255)
                else:
                    # @TODO: generalise this to any W, H and n_channels
                    data = np.array(data).reshape((1, 256, 256, 3))
                    pred = self.model.predict(data).reshape((1, 256, 256))
                    pred *= 255
                    pred = pred.astype(np.uint8)
                    cv2.imwrite(os.path.join(self.pred_path, "{}-img.{}[{}].png"
                                             .format(data_type, epoch+1, self.run_id)), pred[0]*255)


