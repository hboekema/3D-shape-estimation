import os
import argparse
import json
from datetime import datetime
from keras.models import Model

import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf

from encoder import EncoderArchitecture
from smpl_np import SMPLModel
from render_mesh import Mesh
from callbacks import PredOnEpochEnd
from silhouette_generator import SilhouetteDataGenerator


""" Set-up """

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to configuration file")
parser.add_argument("--silhouettes", help="Path to pre-generated silhouettes")
parser.add_argument("--run_id", help="Identifier of this network pass")

args = parser.parse_args()

# Read in the configurations
if args.config is not None:
    with open(args.config, 'r') as file:
        params = json.load(file)
else:
    with open("../config/model_config.json", 'r') as file:
        params = json.load(file)

# Set the ID of this training pass
if args.run_id is not None:
    run_id = args.run_id
else:
    # Use the current date and time as a run-id
    run_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Create experiment directory
exp_dir = "../experiments/" + str(run_id) + "/"
model_dir = exp_dir + "models/"
logs_dir = exp_dir + "logs/"
train_vis_dir = exp_dir + "train_vis/"
test_vis_dir = exp_dir + "test_vis/"
os.mkdir(exp_dir)
os.mkdir(model_dir)
os.mkdir(logs_dir)
os.mkdir(train_vis_dir)
os.mkdir(test_vis_dir)

# Set number of GPUs to use
os.environ["CUDA_VISIBLE_DEVICES"] = params["ENV"]["CUDA_VISIBLE_DEVICES"]

# Ensure that TF2.0 is not used
tf.disable_v2_behavior()
tf.enable_eager_execution()

# Set Keras format
tf.keras.backend.set_image_data_format(params["ENV"]["CHANNEL_FORMAT"])

# Store the data paths
train_dir = params["DATA"]["SOURCE"]["TRAIN"]
val_dir = params["DATA"]["SOURCE"]["VAL"]
test_dir = params["DATA"]["SOURCE"]["TEST"]

# Store the width, height and number of channels of the silhouettes
silh_dim = params["DATA"]["SILH_INFO"]["INPUT_WH"]
silh_n_channels = params["DATA"]["SILH_INFO"]["N_CHANNELS"]

# Store the batch size and number of epochs
batch_size = params["GENERATOR"]["BATCH_SIZE"]
epochs = params["MODEL"]["EPOCHS"]
steps_per_epoch = params["MODEL"]["STEPS_PER_EPOCH"]
validation_steps = params["MODEL"]["VALIDATION_STEPS"]

""" Data generation """

# Load the SMPL data
train_gen = SilhouetteDataGenerator(train_dir, batch_size=batch_size, img_dim=silh_dim, frac_randomised=1.0)
val_gen = SilhouetteDataGenerator(val_dir, batch_size=batch_size, img_dim=silh_dim, frac_randomised=1.0)
test_gen = SilhouetteDataGenerator(test_dir, batch_size=batch_size, img_dim=silh_dim, frac_randomised=1.0, noise=0.0)


# Generate the silhouettes from the SMPL parameters
smpl = SMPLModel('../SMPL/model.pkl')

# Artificial sample data
sample_pose = 0.65 * (np.random.rand(*smpl.pose_shape) - 0.5)
sample_beta = 0.06 * (np.random.rand(*smpl.beta_shape) - 0.5)
sample_trans = np.zeros(smpl.trans_shape)

sample_y = np.array([sample_pose.ravel(), sample_beta, sample_trans])
sample_pc = smpl.set_params(sample_pose, sample_beta, sample_trans)
sample_x = Mesh(pointcloud=sample_pc).render_silhouette(dim=silh_dim, show=False)
sample_x = sample_x.reshape((*silh_dim, silh_n_channels)).astype("float32")
sample_x /= 255

""" Model set-up"""

# Callback functions
# Create a model checkpoint after every few epochs
model_save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_dir + "model.{epoch:02d}-{loss:.2f}.hdf5",
    monitor='loss', verbose=1, save_best_only=False, mode='auto',
    period=params["MODEL"]["CHKPT_PERIOD"])

# Predict on sample images at the end of every few epochs
epoch_pred_cb = PredOnEpochEnd(logs_dir, smpl, x_test=sample_x,
                               pred_path=train_vis_dir, period=params["MODEL"]["CHKPT_PERIOD"], visualise=True)


# Make model entity
encoder_inputs, encoder_outputs = EncoderArchitecture((*silh_dim, silh_n_channels))
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_outputs)
encoder_model.summary()

encoder_model.compile(optimizer='adam', loss=['mse', mesh_mse], loss_weights=[1.0, 1.0])
epoch_pred_cb.set_model(encoder_model)

""" Training loop"""

# Run the main loop
encoder_model.fit_generator(
    generator=train_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=[epoch_pred_cb, model_save_checkpoint],
    use_multiprocessing=params["ENV"]["USE_MULTIPROCESSING"]
)

# Store the model
encoder_model.save(model_dir + "model.final.hdf5")

