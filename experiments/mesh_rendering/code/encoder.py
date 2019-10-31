import numpy as np
import tensorflow.compat.v1 as tf

from smpl_tf import smpl_model
from render_mesh import Mesh


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1a = tf.keras.layers.Conv2D(256, (5, 5), activation="relu")
#        self.conv1b = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.25)

        self.conv2a = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
#        self.conv2b = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.25)

        self.conv3a = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
#        self.conv3b = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.dropout3 = tf.keras.layers.Dropout(0.25)

        self.conv4a = tf.keras.layers.Conv2D(64, (7, 7), activation="relu")
#        self.conv4b = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
        self.pool4 = tf.keras.layers.MaxPooling2D((2, 2))
        self.batchnorm4 = tf.keras.layers.BatchNormalization()
        self.dropout4 = tf.keras.layers.Dropout(0.25)

        self.conv5a = tf.keras.layers.Conv2D(64, (7, 7), activation="relu")
#        self.conv5b = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
        self.pool5 = tf.keras.layers.MaxPooling2D((2, 2))
        self.batchnorm5 = tf.keras.layers.BatchNormalization()
        self.dropout5 = tf.keras.layers.Dropout(0.25)

        self.conv6a = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")
#        self.conv6b = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")
        self.batchnorm6 = tf.keras.layers.BatchNormalization()
        self.dropout6 = tf.keras.layers.Dropout(0.5)

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")
        self.dropout4 = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(85, activation="tanh")

        self.loss_fn = self.mesh_mse
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = 0.0
        self.iou = 0.0

        self.loss = tf.keras.metrics.Mean(name='train_loss')
        self.accuracy = tf.keras.metrics.Mean(name='train_iou')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Mean(name='val_iou')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.Mean(name='test_iou')

    def call(self, x):
        x = self.conv1a(x)
        #x = self.conv1b(x)
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = self.dropout1(x)

        x = self.conv2a(x)
        #x = self.conv2b(x)
        x = self.pool2(x)
        x = self.batchnorm2(x)
        x = self.dropout2(x)

        x = self.conv3a(x)
        #x = self.conv3b(x)
        x = self.pool3(x)
        x = self.batchnorm3(x)
        x = self.dropout3(x)

        x = self.conv4a(x)
        #x = self.conv4b(x)
        x = self.pool4(x)
        x = self.batchnorm4(x)
        x = self.dropout4(x)

        x = self.conv5a(x)
        #x = self.conv5b(x)
        x = self.pool5(x)
        x = self.batchnorm5(x)
        x = self.dropout5(x)

        x = self.conv6a(x)
        #x = self.conv6b(x)
        x = self.batchnorm6(x)
        x = self.dropout6(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        x = self.dropout4(x)
        x = self.dense3(x)

        return x

    def mesh_mse(self, y_true, y_pred):
        """ Calculate the Euclidean distance between pairs of vertices in true and predicted point cloud,
        generated from SMPL parameters """
        # Cast the arrays to 64-bit float
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)

        # Calculate Euclidean distances between nearest points
        mesh_dists = []
        iou = []
        for i in range(len(y_true)):
            # Generate meshes from the SMPL parameters
            pc_true, _ = smpl_model('../SMPL/model.pkl', y_true[i, 72:82], y_true[i, :72], y_true[i, 82:85])
            pc_pred, _ = smpl_model('../SMPL/model.pkl', y_pred[i, 72:82], y_pred[i, :72], y_pred[i, 82:85])

            # Apply nearest-neighbours (using Euclidean distances) to evaluate the mesh's mse @TODO: nearest-neighbour
            distance = tf.math.reduce_mean(tf.math.square(tf.add(pc_true, tf.negative(pc_pred))))
            mesh_dists.append(distance)

            # Store silhouettes for IOU evaluation metric
            # Generate silhouettes from the point clouds
            img_dim = (256, 256)
            silh_true = Mesh(pointcloud=pc_true.numpy()).render_silhouette(dim=img_dim, show=False)
            silh_pred = Mesh(pointcloud=pc_pred.numpy()).render_silhouette(dim=img_dim, show=False)

            silh_true = np.divide(silh_true, 255).astype(np.bool)
            silh_pred = np.divide(silh_pred, 255).astype(np.bool)

            # Use Boolean algebra to calculate the intersection and union of the silhouettes
            intersection = silh_true * silh_pred
            union = silh_true + silh_pred

            iou.append(intersection.sum()/float(union.sum()))

        # Calculate the mean sihouette IOU score
        self.iou = np.mean(iou)

        # Compute the mean of the mse over all of the predictions in this batch
        return tf.math.reduce_mean(mesh_dists)

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.call(x)
            loss = self.loss_fn(tf.convert_to_tensor(y), predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss(loss)
        self.accuracy(self.iou)

    def val_step(self, x, y):
        predictions = self.call(x)
        val_loss = self.loss_fn(tf.convert_to_tensor(y), predictions)

        self.val_loss(val_loss)
        self.val_accuracy(self.iou)

    def predict_(self, x):
        """ Wrapper for predictions """
        return np.array(self.call(x))
