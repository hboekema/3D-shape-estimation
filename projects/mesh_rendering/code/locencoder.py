import sys
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Conv2D, Lambda, Concatenate, Dropout, BatchNormalization, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19

sys.path.append('/data/cvfs/hjhb2/projects/mesh_rendering/code/keras_rotationnet_v2_demo_for_hidde/')
sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from points3d import Points3DFromSMPLParams, get_parameters
from smpl_np_rot_v6 import load_params
#from smpl_tf import smpl_model, smpl_model_batched
from render_mesh import Mesh


#class LocEncoder():
#    def __init__(self, input_shape=(256, 256), pc_shape=(6890, 3)):
#        self.input_shape = input_shape
#        self.pc_shape = pc_shape
#
#        self.encoder_model = self.create_encoder(self.input_shape)
#        self.loclearner_model = self.create_loclearner(self.pc_shape)
#
#    def create_encoder(self, input_shape):
#        encoder_inputs, encoder_outputs = EncoderArchitecture(input_shape)
#        encoder_model = Model(inputs=encoder_inputs, outputs=encoder_outputs)
#        encoder_model.summary()
#
#        encoder_model.compile(optimizer=Adam(), loss=['mse', mesh_mse], loss_weights=[1.0, 1.0])
#        return encoder_model
#
#    def create_loclearner(self, input_shape):
#        loclearner_inputs, loclearner_outputs = LoclearnerArchitecture(input_shape)
#        loclearner_model = Model(inputs=loclearner_inputs, outputs=loclearner_outputs)
#        loclearner_model.summary()
#
#        loclearner_model.compile(optimizer=Adam(), loss="mse")
#        return loclearner_model
#
#    def create_locencoder(self):
#        locencoder_input = Input(shape=self.input_shape)
#        encoder_outputs = self.encoder_model(locencoder_input)
#
#        # Location learner compares GT point cloud with predicted point cloud
#        GT_pointcloud = Input(shape=self.pc_shape)
#        loclearner_outputs = self.loclearner_model(encoder_outputs[1], GT_pointcloud)
#
#        locencoder_model = Model(inputs=[locencoder_input, GT_pointcloud], outputs=loclearner_outputs)
#        locencoder_model.summary()
#
#        locencoder_model.compile(optimizer=Adam())
#        return locencoder_model


# Define custom loss function
def mesh_mse(y_true, y_pred):
    """ Obtain the Euclidean difference between corresponding points in two point clouds """
    return tf.reduce_mean(tf.square(tf.add(y_true, tf.negative(y_pred))))


def EncoderArchitecture(input_shape):
    """ Specify the encoder's network architecture """
    encoder_inputs = Input(shape=input_shape)
    #encoder_architecture = encoder_inputs
    #vgg19 = VGG19(include_top=False, input_shape=input_shape, weights=None)
    #for layer in vgg19.layers:
    #    encoder_architecture = layer(encoder_architecture)

    # Network architecture (VGG19)
    # Block 1
    encoder_architecture = Conv2D(64, (3, 3), padding="same", activation="relu")(encoder_inputs)
    encoder_architecture = Conv2D(64, (3, 3), padding="same", activation="relu")(encoder_architecture)
    encoder_architecture = BatchNormalization()(encoder_architecture)
    encoder_architecture = MaxPooling2D((2, 2))(encoder_architecture)
    encoder_architecture = Dropout(0.25)(encoder_architecture)

    # Block 2
    encoder_architecture = Conv2D(128, (3, 3), padding="same", activation="relu")(encoder_architecture)
    encoder_architecture = Conv2D(128, (3, 3), padding="same", activation="relu")(encoder_architecture)
    encoder_architecture = BatchNormalization()(encoder_architecture)
    encoder_architecture = MaxPooling2D((2, 2))(encoder_architecture)
    encoder_architecture = Dropout(0.25)(encoder_architecture)

    # Block 3
    encoder_architecture = Conv2D(256, (3, 3), padding="same", activation="relu")(encoder_architecture)
    encoder_architecture = Conv2D(256, (3, 3), padding="same", activation="relu")(encoder_architecture)
    #encoder_architecture = Conv2D(256, (3, 3), padding="same", activation="relu")(encoder_architecture)
    encoder_architecture = BatchNormalization()(encoder_architecture)
    encoder_architecture = MaxPooling2D((2, 2))(encoder_architecture)
    encoder_architecture = Dropout(0.25)(encoder_architecture)

    # Block 4
    encoder_architecture = Conv2D(256, (3, 3), activation="relu")(encoder_architecture)
    encoder_architecture = Conv2D(256, (3, 3), activation="relu")(encoder_architecture)
    #encoder_architecture = Conv2D(256, (3, 3), activation="relu")(encoder_architecture)
    encoder_architecture = BatchNormalization()(encoder_architecture)
    encoder_architecture = MaxPooling2D((2, 2))(encoder_architecture)
    encoder_architecture = Dropout(0.25)(encoder_architecture)

    # Block 5
    encoder_architecture = Conv2D(512, (3, 3), activation="relu")(encoder_architecture)
    encoder_architecture = Conv2D(512, (3, 3), activation="relu")(encoder_architecture)
    #encoder_architecture = Conv2D(512, (3, 3), activation="relu")(encoder_architecture)
    encoder_architecture = BatchNormalization()(encoder_architecture)
    encoder_architecture = MaxPooling2D((2, 2))(encoder_architecture)
    encoder_architecture = Dropout(0.25)(encoder_architecture)

    # Block 6
    encoder_architecture = Conv2D(512, (3, 3), activation="relu")(encoder_architecture)
    #encoder_architecture = Conv2D(512, (3, 3), padding="same", activation="relu")(encoder_architecture)
    #encoder_architecture = Conv2D(512, (3, 3), padding="same", activation="relu")(encoder_architecture)
    encoder_architecture = BatchNormalization()(encoder_architecture)
    encoder_architecture = AveragePooling2D((3, 3))(encoder_architecture)
    encoder_architecture = Dropout(0.25)(encoder_architecture)

    # Dense layers
    encoder_architecture = Flatten()(encoder_architecture)
    encoder_architecture = Dense(1024)(encoder_architecture)
    encoder_architecture = Dropout(0.5)(encoder_architecture)
    encoder_architecture = Dense(512)(encoder_architecture)
    encoder_architecture = Dropout(0.5)(encoder_architecture)

    encoder_params = Dense(85, activation="tanh")(encoder_architecture)

    # Load SMPL model and get necessary parameters
    smpl_params = load_params('./keras_rotationnet_v2_demo_for_hidde//basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    _, _, input_info = get_parameters()
    input_betas = Lambda(lambda x: x[:, 72:82])(encoder_params)
    input_pose_rodrigues = Lambda(lambda x: x[:, 0:72])(encoder_params)
    input_trans = Lambda(lambda x: x[:, 82:85])(encoder_params)

    encoder_mesh = Points3DFromSMPLParams(input_betas, input_pose_rodrigues, input_trans, smpl_params, input_info)
    #encoder_layer = Lambda(Points3DFromSMPLParams, output_shape=(6890, 3))
    #encoder_mesh = encoder_layer([encoder_architecture[:, 72:82], encoder_architecture[:, :72], encoder_architecture[:, 82:85], smpl_params, input_info])
    #encoder_outputs = [encoder_architecture]#, encoder_mesh]
    #encoder_outputs = [encoder_architecture, encoder_mesh]
    #print(encoder_inputs)
    #exit(1)

    return [encoder_inputs], [encoder_params, encoder_mesh]


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

def LocEncoderArchitecture(img_shape, mesh_shape=(6890, 3)):
    """ Localised-learning encoder """
    encoder_input = Input(shape=img_shape)
    encoder_architecture = Conv2D(32, (3, 3), padding="same", activation="relu")(encoder_input)
    encoder_architecture = Flatten()(encoder_architecture)
    encoder_architecture = Dense(85)(encoder_architecture)
    encoder_mesh = Points3DFromSMPLParams(encoder_architecture[:, 72:82], encoder_architecture[:, :72], encoder_architecture[:, 82:85], smpl_params)

    loclearner_input = Input(shape=mesh_shape)
    flattened_input1 = Flatten(loclearner_input)
    flattened_input2 = Flatten(encoder_mesh)

    concat_inputs = Concatenate()([flattened_input1, flattened_input2])
    loclearner_architecture = Dense(1024)(concat_inputs)
    loclearner_output = Dense(3)(loclearner_architecture)

    return [encoder_input, loclearner_input], [encoder_architecture, encoder_mesh, loclearner_output]



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
