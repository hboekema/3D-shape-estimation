import os
import numpy as np
import tensorflow as tf

from smpl_np import SMPLModel
from render_mesh import Mesh


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
