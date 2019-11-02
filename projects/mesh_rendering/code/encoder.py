import numpy as np
import tensorflow.compat.v1 as tf
from keras.layers import Input, Dense, Flatten, Conv2D, Lambda, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from smpl_tf import smpl_model, smpl_model_batched
from render_mesh import Mesh



class LocEncoder():
    def __init__(self, input_shape=(256, 256), pc_shape=(6890, 3)):
        self.input_shape = input_shape
        self.pc_shape = pc_shape

        self.encoder_model = self.create_encoder(self.input_shape)
        self.loclearner_model = self.create_loclearner(self.pc_shape)

    def create_encoder(self, input_shape):
        encoder_inputs, encoder_outputs = EncoderArchitecture(input_shape)
        encoder_model = Model(inputs=encoder_inputs, outputs=encoder_outputs)
        encoder_model.summary()

        encoder_model.compile(optimizer=Adam(), loss=['mse', mesh_mse], loss_weights=[1.0, 1.0])
        return encoder_model

    def create_loclearner(self, input_shape):
        loclearner_inputs, loclearner_outputs = LoclearnerArchitecture(input_shape)
        loclearner_model = Model(inputs=loclearner_inputs, outputs=loclearner_outputs)
        loclearner_model.summary()

        loclearner_model.compile(optimizer=Adam(), loss="mse")
        return loclearner_model

    def create_locencoder(self):
        locencoder_input = Input(shape=self.input_shape)
        encoder_outputs = self.encoder_model(locencoder_input)

        # Location learner compares GT point cloud with predicted point cloud
        GT_pointcloud = Input(shape=self.pc_shape)
        loclearner_outputs = self.loclearner_model(encoder_outputs[1], GT_pointcloud)

        locencoder_model = Model(inputs=[locencoder_input, GT_pointcloud], outputs=loclearner_outputs)
        locencoder_model.summary()

        locencoder_model.compile(optimizer=Adam())
        return locencoder_model


# Define custom loss function
def mesh_mse(y_true, y_pred):
    """ Obtain the Euclidean difference between corresponding points in two point clouds """
    return tf.math.reduce_mean(tf.math.square(tf.add(y_true, tf.negative(y_pred))))


def EncoderArchitecture(input_shape):
    """ Specify the encoder's network architecture """
    encoder_inputs = [Input(shape=input_shape)]
    encoder_architecture = Conv2D(32, (3, 3), padding="same", activation="relu")(encoder_inputs[0])
    encoder_architecture = Flatten()(encoder_architecture)
    encoder_architecture = Dense(85)(encoder_architecture)
    encoder_parameters = tf.cast(encoder_architecture, tf.float64)
    encoder_mesh = Lambda(smpl_model("../model.pkl", encoder_parameters[:, 72:82], encoder_parameters[:, :72],
                                             encoder_parameters[:, 82:85]), output_shape=(6890, 3))
    encoder_outputs = [encoder_architecture, encoder_mesh]

    return encoder_inputs, encoder_outputs


def LoclearnerArchitecture(input_shape=(6890, 3)):
    """ Build architecure for localised learning network """
    loclearner_input1 = Input(shape=input_shape)
    loclearner_input2 = Input(shape=input_shape)
    flattened_input1 = Flatten(loclearner_input1)
    flattened_input2 = Flatten(loclearner_input2)

    concat_inputs = Concatenate()([flattened_input1, flattened_input2])

    loclearner_architecture = Dense(1024)(concat_inputs)
    loclearner_outputs = [Dense(3)(loclearner_architecture)]

    return [loclearner_input1, loclearner_input2], loclearner_outputs





# class Encoder(tf.keras.Model):
#     def __init__(self, img_dim=(256, 256)):
#         super(Encoder, self).__init__()
#         self.conv1a = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=img_dim)
#         self.conv1b = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")
#         self.batchnorm1 = tf.keras.layers.BatchNormalization()
#         self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
#         self.dropout1 = tf.keras.layers.Dropout(0.25)
#
#         self.conv2a = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")
#         #self.conv2b = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
#         self.batchnorm2 = tf.keras.layers.BatchNormalization()
#         self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
#         self.dropout2 = tf.keras.layers.Dropout(0.25)
#
#         self.conv3a = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")
#         # self.conv3b = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
#         self.batchnorm3 = tf.keras.layers.BatchNormalization()
#         self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
#         self.dropout3 = tf.keras.layers.Dropout(0.25)
#
#         self.conv4a = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
#         self.conv4b = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
#         self.batchnorm4 = tf.keras.layers.BatchNormalization()
#         self.pool4 = tf.keras.layers.MaxPooling2D((2, 2))
#         self.dropout4 = tf.keras.layers.Dropout(0.25)
#
#         self.conv5a = tf.keras.layers.Conv2D(256, (3, 3), activation="relu")
#         # self.conv5b = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
#         self.batchnorm5 = tf.keras.layers.BatchNormalization()
#         self.pool5 = tf.keras.layers.MaxPooling2D((2, 2))
#         self.dropout5 = tf.keras.layers.Dropout(0.25)
#
#         self.conv6a = tf.keras.layers.Conv2D(256, (3, 3), activation="relu")
#         # self.conv6b = tf.keras.layers.Conv2D(256, (3, 3), activation="relu")
#         self.batchnorm6 = tf.keras.layers.BatchNormalization()
#         self.pool6 = tf.keras.layers.AveragePooling2D((3, 3))
#         self.dropout6 = tf.keras.layers.Dropout(0.25)
#
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense1 = tf.keras.layers.Dense(512, activation="relu")
#         self.dropout3 = tf.keras.layers.Dropout(0.5)
#         self.dense2 = tf.keras.layers.Dense(256, activation="relu")
#         self.dropout4 = tf.keras.layers.Dropout(0.5)
#         self.dense3 = tf.keras.layers.Dense(85, activation="tanh")
#
#         self.loss_fn = self.mesh_mse
#         self.optimizer = tf.keras.optimizers.Adam()
#         self.loss_value = 0.0
#         self.iou = 0.0
#
#         self.loss = tf.keras.metrics.Mean(name='train_loss')
#         self.accuracy = tf.keras.metrics.Mean(name='train_iou')
#
#         self.val_loss = tf.keras.metrics.Mean(name='val_loss')
#         self.val_accuracy = tf.keras.metrics.Mean(name='val_iou')
#
#         self.test_loss = tf.keras.metrics.Mean(name='test_loss')
#         self.test_accuracy = tf.keras.metrics.Mean(name='test_iou')
#
#     def call(self, x):
#         x = self.conv1a(x)
#         x = self.conv1b(x)
#         x = self.batchnorm1(x)
#         x = self.pool1(x)
#         x = self.dropout1(x)
#
#         x = self.conv2a(x)
#         # x = self.conv2b(x)
#         x = self.batchnorm2(x)
#         x = self.pool2(x)
#         x = self.dropout2(x)
#
#         x = self.conv3a(x)
#         # x = self.conv3b(x)
#         x = self.batchnorm3(x)
#         x = self.pool3(x)
#         x = self.dropout3(x)
#
#         x = self.conv4a(x)
#         x = self.conv4b(x)
#         x = self.batchnorm4(x)
#         x = self.pool4(x)
#         x = self.dropout4(x)
#
#         x = self.conv5a(x)
#         # x = self.conv5b(x)
#         x = self.batchnorm5(x)
#         x = self.pool5(x)
#         x = self.dropout5(x)
#
#         x = self.conv6a(x)
#         # x = self.conv6b(x)
#         x = self.batchnorm6(x)
#         x = self.pool6(x)
#         x = self.dropout6(x)
#
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.dropout3(x)
#         x = self.dense2(x)
#         x = self.dropout4(x)
#         x = self.dense3(x)
#
#         return x
#
#     def mesh_mse(self, y_true, y_pred):
#         """ Calculate the Euclidean distance between pairs of vertices in true and predicted point cloud,
#         generated from SMPL parameters """
#         # Cast the arrays to 64-bit float
#         y_true = tf.cast(y_true, tf.float64)
#         y_pred = tf.cast(y_pred, tf.float64)
#
#         # Calculate Euclidean distances between nearest points
#         mesh_dists = []
#         iou = []
#         for i in range(len(y_true)):
#             # Generate meshes from the SMPL parameters
#             pc_true, _ = smpl_model('../SMPL/model.pkl', y_true[i, 72:82], y_true[i, :72], y_true[i, 82:85])
#             pc_pred, _ = smpl_model('../SMPL/model.pkl', y_pred[i, 72:82], y_pred[i, :72], y_pred[i, 82:85])
#
#             # Apply nearest-neighbours (using Euclidean distances) to evaluate the mesh's mse @TODO: nearest-neighbour
#             distance = tf.math.reduce_mean(tf.math.square(tf.add(pc_true, tf.negative(pc_pred))))
#             mesh_dists.append(distance)
#
#             # Store silhouettes for IOU evaluation metric
#             # Generate silhouettes from the point clouds
#             img_dim = (256, 256)
#             silh_true = Mesh(pointcloud=pc_true.numpy()).render_silhouette(dim=img_dim, show=False)
#             silh_pred = Mesh(pointcloud=pc_pred.numpy()).render_silhouette(dim=img_dim, show=False)
#
#             silh_true = np.divide(silh_true, 255).astype(np.bool)
#             silh_pred = np.divide(silh_pred, 255).astype(np.bool)
#
#             # Use Boolean algebra to calculate the intersection and union of the silhouettes
#             intersection = silh_true * silh_pred
#             union = silh_true + silh_pred
#
#             iou.append(intersection.sum()/float(union.sum()))
#
#         # Calculate the mean sihouette IOU score
#         self.iou = np.mean(iou)
#
#         # Compute the mean of the mse over all of the predictions in this batch
#         return tf.math.reduce_mean(mesh_dists)
#
#     @staticmethod
#     def global_mesh_mse(y_true, y_pred):
#         """ Calculate the Euclidean distance between pairs of vertices in true and predicted point cloud,
#         generated from SMPL parameters """
#         # Cast the arrays to 64-bit float
#         y_true = tf.cast(y_true, tf.float64)
#         y_pred = tf.cast(y_pred, tf.float64)
#
#         # Calculate Euclidean distances between nearest points
#         mesh_dists = []
#         iou = []
#         for i in range(len(y_true)):
#             # Generate meshes from the SMPL parameters
#             pc_true, _ = smpl_model('../SMPL/model.pkl', y_true[i, 72:82], y_true[i, :72], y_true[i, 82:85])
#             pc_pred, _ = smpl_model('../SMPL/model.pkl', y_pred[i, 72:82], y_pred[i, :72], y_pred[i, 82:85])
#
#             # Apply nearest-neighbours (using Euclidean distances) to evaluate the mesh's mse @TODO: nearest-neighbour
#             distance = tf.math.reduce_mean(tf.math.square(tf.add(pc_true, tf.negative(pc_pred))))
#             mesh_dists.append(distance)
#
#         #     # Store silhouettes for IOU evaluation metric
#         #     # Generate silhouettes from the point clouds
#         #     img_dim = (256, 256)
#         #     silh_true = Mesh(pointcloud=pc_true.numpy()).render_silhouette(dim=img_dim, show=False)
#         #     silh_pred = Mesh(pointcloud=pc_pred.numpy()).render_silhouette(dim=img_dim, show=False)
#         #
#         #     silh_true = np.divide(silh_true, 255).astype(np.bool)
#         #     silh_pred = np.divide(silh_pred, 255).astype(np.bool)
#         #
#         #     # Use Boolean algebra to calculate the intersection and union of the silhouettes
#         #     intersection = silh_true * silh_pred
#         #     union = silh_true + silh_pred
#         #
#         #     iou.append(intersection.sum() / float(union.sum()))
#         #
#         # # Calculate the mean sihouette IOU score
#         # self.iou = np.mean(iou)
#
#         # Compute the mean of the mse over all of the predictions in this batch
#         return tf.math.reduce_mean(mesh_dists)
#
#     def train_step(self, x, y):
#         with tf.GradientTape() as tape:
#             predictions = self.call(x)
#             loss = self.loss_fn(tf.convert_to_tensor(y), predictions)
#
#         gradients = tape.gradient(loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
#
#         self.loss_value = loss
#         self.loss(loss)
#         self.accuracy(self.iou)
#
#     def val_step(self, x, y):
#         predictions = self.call(x)
#         val_loss = self.loss_fn(tf.convert_to_tensor(y), predictions)
#
#         self.val_loss(val_loss)
#         self.val_accuracy(self.iou)
#
#     def predict_(self, x):
#         """ Wrapper for predictions """
#         return np.array(self.call(x))
