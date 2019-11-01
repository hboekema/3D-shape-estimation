import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import collections

# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to configuration file")
parser.add_argument("--silhouettes", help="Path to pre-generated silhouettes")
parser.add_argument("--run_id", help="Identifier of this network pass")

args = parser.parse_args()

# Read in the configurations
with open(args.config, 'r') as file:
    params = json.load(file)

# Set the ID of this training pass
if args.run_id is not None:
    run_id = args.run_id
else:
    # Use the current date and time as a run-id
    run_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Set number of GPUs to use
os.environ["CUDA_VISIBLE_DEVICES"] = params["ENV"]["CUDA_VISIBLE_DEVICES"]

import glob
import numpy as np
import tensorflow.compat.v1 as tf
#import tensorflow as tf

from encoder import Encoder
from smpl_np import SMPLModel
from render_mesh import Mesh
from callbacks import PredOnEpochEnd

# Ensure that TF2.0 is not used
tf.disable_v2_behavior()
tf.enable_eager_execution()
print(tf.executing_eagerly())

# Set Keras format
tf.keras.backend.set_image_data_format(params["ENV"]["CHANNEL_FORMAT"])

# Store the data paths
train_dir = params["DATA"]["SOURCE"]["TRAIN"]
val_dir = params["DATA"]["SOURCE"]["VAL"]
test_dir = params["DATA"]["SOURCE"]["TEST"]

# Store the width, height and number of channels of the silhouettes
silh_dim = params["DATA"]["SILH_INFO"]["INPUT_WH"]
silh_n_channels = params["DATA"]["SILH_INFO"]["N_CHANNELS"]

# Store the salient paths
log_path = params["ENV"]["LOG_PATH"]
train_pred_path = params["ENV"]["TRAINING_VIS_PATH"]
test_pred_path = params["ENV"]["TEST_VIS_PATH"]

# Store the batch size and number of epochs
batch_size = params["GENERATOR"]["BATCH_SIZE"]
epochs = params["MODEL"]["EPOCHS"]
steps_per_epoch = params["MODEL"]["STEPS_PER_EPOCH"]
validation_steps = params["MODEL"]["VALIDATION_STEPS"]


# Load the SMPL data
class SilhouetteDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=32, img_dim=(256, 256), frac_randomised=0.2, noise=0.01):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.frac_randomised = frac_randomised  # fraction of parameters to generate randomly in each batch
        self.noise = noise

    def __call__(self, *args, **kwargs):
        return self.__next__()

    def __len__(self):
        return int(np.ceil(len(os.listdir(self.data_dir)) / self.batch_size))

    def __getitem__(self, item):
        """ Yield batches of data """
        # Load the SMPL model
        smpl = SMPLModel('../SMPL/model.pkl')

        # Split of random and real data
        num_artificial = int(np.round(self.frac_randomised * self.batch_size))
        num_real = int(self.batch_size - num_artificial)

        # Retrieve a random batch of parameters from the data directory
        if num_real > 0:
            data = np.array(os.listdir(self.data_dir))
            Y_batch_ids = data[np.random.randint(low=0, high=data.shape[0], size=num_real)]
        else:
            Y_batch_ids = []

        Y_batch = []
        X_batch = []
        for Y_id in Y_batch_ids:
            # Fetch the real data
            Y = np.load(os.path.join(self.data_dir, Y_id))

            # Add a small amount of noise to the data
            Y += np.random.uniform(low=-self.noise, high=self.noise, size=Y.shape)
            Y_batch.append(Y)

            # Now generate the silhouette from the SMPL meshes
            # Create the body mesh
            pose = Y[:72]
            beta = Y[72:82]
            trans = Y[82:]
            pointcloud = smpl.set_params(pose.reshape((24, 3)), beta, trans)

            # Render the silhouette
            silhouette = Mesh(pointcloud=pointcloud).render_silhouette(dim=self.img_dim, show=False)
            X_batch.append(np.array(silhouette))

        for i in range(len(range(num_artificial))):
            # Generate artificial data
            pose = 0.65 * (np.random.rand(*smpl.pose_shape) - 0.5)
            beta = 0.06 * (np.random.rand(*smpl.beta_shape) - 0.5)
            trans = np.zeros(smpl.trans_shape)

            Y_batch.append(np.concatenate([pose.ravel(), beta, trans]))

            # Create the body mesh
            pointcloud = smpl.set_params(beta=beta, pose=pose, trans=trans)

            # Render the silhouette
            silhouette = Mesh(pointcloud=pointcloud).render_silhouette(dim=self.img_dim, show=False)
            X_batch.append(np.array(silhouette))

        # Preprocess the batches and yield them
        Y_batch = np.array(Y_batch, dtype="float32")
        X_batch = np.array(X_batch, dtype="float32")
        X_batch /= 255
        X_batch = X_batch.reshape((*X_batch.shape, 1))

        return X_batch, Y_batch


train_gen = SilhouetteDataGenerator(train_dir, batch_size=batch_size, img_dim=silh_dim, frac_randomised=1.0)
val_gen = SilhouetteDataGenerator(val_dir, batch_size=batch_size, img_dim=silh_dim, frac_randomised=1.0)
test_gen = SilhouetteDataGenerator(test_dir, batch_size=batch_size, img_dim=silh_dim, frac_randomised=1.0, noise=0.0)

# Generate TF Datasets from the generators
train_ds = tf.data.Dataset.from_generator(train_gen, output_types=(np.uint8, np.float32))
val_ds = tf.data.Dataset.from_generator(val_gen, output_types=(np.uint8, np.float32))
test_ds = tf.data.Dataset.from_generator(test_gen, output_types=(np.uint8, np.float32))


# Samples for evaluating the model after epochs
# train_sample_y = np.array(np.load(os.path.join(train_dir, "train_sample_0000.npy")))
# test_sample_y = np.array(np.load(os.path.join(test_dir, "test_sample_0000.npy")))

# Generate the silhouettes from the SMPL parameters
smpl = SMPLModel('../SMPL/model.pkl')
# train_sample_pc = smpl.set_params(train_sample_y[:72].reshape((24, 3)), train_sample_y[72:82], train_sample_y[82:])
# test_sample_pc = smpl.set_params(test_sample_y[:72].reshape((24, 3)), test_sample_y[72:82], test_sample_y[82:])
# train_sample = Mesh(pointcloud=train_sample_pc).render_silhouette(dim=silh_dim, show=False)
# test_sample = Mesh(pointcloud=test_sample_pc).render_silhouette(dim=silh_dim, show=False)

# Format the sample data
# train_sample = train_sample.reshape((*silh_dim, silh_n_channels)).astype("float32")
# test_sample = test_sample.reshape((*silh_dim, silh_n_channels)).astype("float32")
# train_sample /= 255
# test_sample /= 255

# Artificial sample data
sample_pose = 0.65 * (np.random.rand(*smpl.pose_shape) - 0.5)
sample_beta = 0.06 * (np.random.rand(*smpl.beta_shape) - 0.5)
sample_trans = np.zeros(smpl.trans_shape)

sample_y = np.array([sample_pose.ravel(), sample_beta, sample_trans])
sample_pc = smpl.set_params(sample_pose, sample_beta, sample_trans)
sample_x = Mesh(pointcloud=sample_pc).render_silhouette(dim=silh_dim, show=False)
sample_x = sample_x.reshape((*silh_dim, silh_n_channels)).astype("float32")
sample_x /= 255

# # Get the SMPL data
# Y_train = []
# print("Loading data...")
# for np_name in glob.glob(os.path.join(data_dir, '*.np[yz]')):
#     data = np.load(np_name)
#
#     # Get the poses
#     poses = data["poses"]
#     poses = poses.reshape(poses.shape[0], 52, 3)[:, :24, :]
#     # Get the betas
#     betas = data["betas"][:10]
#     # Get the trans
#     trans = data["trans"]
#
#     # Concatenate the data to fit in a single array
#     for i in range(poses.shape[0]):
#         Y_train.append(np.concatenate([poses[i].ravel(), betas, trans[i]]))
#
# Y_train = np.array(Y_train, dtype="float32")
#
# if args.silhouettes is not None:
#     print("Fetching silhouettes...")
#     # Load the silhouettes
#     X_train = np.array([cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(args.silhouettes + "*.png")], dtype="float32")
# else:
#     print("Generating silhouettes...")
#     X_train = []
#     # Generate the silhouettes from the SMPL parameters
#     smpl = SMPLModel('../SMPL/model.pkl')
#     silh_path = "../silhouettes/"
#     for i, params in enumerate(Y_train):
#         pose = params[:72].reshape((24, 3))
#         beta = params[72:82]
#         trans = params[82:]
#
#         # Create the body mesh
#         pointcloud = smpl.set_params(pose, beta, trans)
#
#         # Render the silhouette
#         silhouette = Mesh(pointcloud=pointcloud).render2D(save_path="../silhouettes/silhouette_{:03d}.png".format(i))
#         X_train.append(np.array(silhouette))
#
#     X_train = np.array(X_train, dtype="float32")
#
# X_train /= 255
# X_train = X_train.reshape((*X_train.shape, 1))
#
# # Split the dataset into train, test and validation sets
# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1)
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)

# Define loss function
def mesh_mse(y_true, y_pred):
    """ Calculate the Euclidean distance between pairs of vertices in true and predicted point cloud,
    generated from SMPL parameters """
    # Specify silhouette dimensions
    img_dim = (256, 256)

    # Cast the arrays to 64-bit float
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    # y_true = tf.cast(y_true, tf.float64)
    # y_pred = tf.cast(y_pred, tf.float64)

    # Generate meshes from the SMPL parameters
    # pc_true, _ = smpl_model('../SMPL/model.pkl', y_true[i, 72:82], y_true[i, :72], y_true[i, 82:85])
    # pc_pred, _ = smpl_model('../SMPL/model.pkl', y_pred[72:82], y_pred[:72], y_pred[82:85])
    pc_true = smpl.set_params(y_true[:72], y_true[72:82], y_true[82:])
    pc_pred = smpl.set_params(y_pred[:72], y_pred[72:82], y_pred[82:])

    # Generate silhouettes from the point clouds
    silh_true = Mesh(pointcloud=pc_true).render_silhouette(dim=img_dim, show=False)
    silh_pred = Mesh(pointcloud=pc_pred).render_silhouette(dim=img_dim, show=False)

    # Return the silhouette's cross-entropy
    return tf.keras.losses.BinaryCrossentropy(silh_true, silh_pred)

# Callback functions
# Create a model checkpoint after every few epochs
# model_save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     "../models/model.{epoch:02d}-{loss:.2f} " + str(run_id) + ".hdf5",
#     monitor='loss', verbose=1, save_best_only=False, mode='auto',
#     period=params["MODEL"]["CHKPT_PERIOD"])

# Predict on sample images at the end of every few epochs
epoch_pred_cb = PredOnEpochEnd(log_path, smpl, x_test=sample_x,
                               pred_path=train_pred_path, run_id=run_id, period=params["MODEL"]["CHKPT_PERIOD"], visualise=False)

# Make model entity
encoder = Encoder()
epoch_pred_cb.set_model(encoder)
#model_save_checkpoint.set_model(encoder)

# Run the main loop
for epoch in range(1, epochs + 1):
    print("\nEpoch {}/{}".format(epoch, epochs))
    loss_dict = collections.OrderedDict()
    with tqdm(total=steps_per_epoch) as pbar:
        for i in range(steps_per_epoch):
            batch = train_gen[i]
            train_x = batch[0]
            train_y = batch[1]

            encoder.train_step(train_x, train_y)

            loss_dict["loss"] = "{:03f}".format(encoder.loss.result())
            loss_dict["iou"] = "{:01f}".format(encoder.accuracy.result())

            pbar.set_postfix(loss_dict, refresh=True)
            pbar.update()

    tqdm.close(pbar)

    for i in range(validation_steps):
        batch = val_gen[i]
        val_x = batch[0]
        val_y = batch[1]

        encoder.val_step(val_x, val_y)

    print("[val_loss={:03f}, val_iou={:01f}]".format(encoder.val_loss.result(), encoder.val_accuracy.result()))

    # Model callbacks
    if epoch % params["MODEL"]["CHKPT_PERIOD"] == 0:
        # Save model
        encoder.save_weights("../models/model.{:02d}-{} {}.h5".format(epoch, loss_dict["loss"], str(run_id)))
    epoch_pred_cb.on_epoch_end(epoch, logs=loss_dict)

    # Reset the metrics for the next epoch
    encoder.loss.reset_states()
    encoder.accuracy.reset_states()
    encoder.val_loss.reset_states()
    encoder.val_accuracy.reset_states()


# with tf.Session() as sess:
#     # Use a sequential CNN to learn SMPL parameters
#     encoder = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(*silh_dim, silh_n_channels)),
#         tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.25),
#
#         tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
#         tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.25),
#
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(256, activation="relu"),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(128, activation="relu"),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(85, activation="tanh")
#     ])
#
#     loss = tf.keras.losses.mean_squared_error
#     optimizer = tf.keras.optimizers.Adam()
#
#     encoder.compile(optimizer=optimizer, loss=mesh_mse, metrics=["accuracy"])
#
#     # Train the model
#     print("Training model...")
#     encoder.fit_generator(
#         generator=train_gen,
#         steps_per_epoch=steps_per_epoch,
#         epochs=epochs,
#         validation_data=val_gen,
#         validation_steps=validation_steps,
#         callbacks=[epoch_pred_cb, model_save_checkpoint],
#         use_multiprocessing=params["ENV"]["USE_MULTIPROCESSING"]
#     )
#
#     # Save the final model
#     encoder.save("../models/final_model[{}].hdf5".format(run_id))
#
#     # Predict on the test data set
#     print("Predicting on test data...")
#     preds = encoder.predict_generator(test_gen, steps=validation_steps)
#     for i, pred in enumerate(preds):
#         np.save("../test_visualisations/pred_{:03d}".format(i), pred)
#         smpl.set_params(pred[:72].reshape((24, 3)), pred[72:82], pred[82:])
#         smpl.save_to_obj(os.path.join(test_pred_path, "test_pred_{:04d}[{}].obj".format(i, run_id)))

print("Done.")
