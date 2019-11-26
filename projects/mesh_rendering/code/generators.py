import os
import numpy as np
import tensorflow as tf
import keras

from smpl_np import SMPLModel
from render_mesh import Mesh


class SilhouetteDataGenerator(keras.utils.Sequence):
    def __init__(self, data_dir, smpl, batch_size=32, img_dim=(256, 256), frac_randomised=0.2, noise=0.01, debug=False):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.frac_randomised = frac_randomised  # fraction of parameters to generate randomly in each batch
        self.noise = noise
        self.smpl = smpl
        self.debug = debug
        self.debug_X = None
        self.debug_Y = None

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(os.listdir(self.data_dir)) / self.batch_size))

#    def __next__(self, *args, **kwargs):
#       return self.next(*args, **kwargs)
#
#    def next(self):
#        """For python 2.x
#        # Returns the next batch.    """
#        return __getitem__(0)

    def __getitem__(self, item):
        """ Yield batches of data """
        # Load the SMPL model
        #smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/./basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        smpl = self.smpl

        if self.debug:
            if self.debug_X is None:
                Y_batch_params = []
                Y_batch_pc = []
                X_batch = []
                for i in range(self.batch_size):
                    # Generate artificial data
                    pose = 0.65 * (np.random.rand(smpl.pose_shape[0], smpl.pose_shape[1]) - 0.5)
                    beta = 0.2 * (np.random.rand(smpl.beta_shape[0]) - 0.5)
                    trans = np.zeros(smpl.trans_shape[0])
                    #trans = 0.1 * (np.random.rand(smpl.trans_shape[0]) - 0.5)

                    # Create the body mesh
                    pointcloud = smpl.set_params(beta=beta, pose=pose, trans=trans)
                    Y_batch_params.append(np.concatenate([pose.ravel(), beta, trans]))
                    Y_batch_pc.append(pointcloud)

                    # Render the silhouette
                    silhouette = Mesh(pointcloud=pointcloud).render_silhouette(dim=self.img_dim, show=False)
                    X_batch.append(np.array(silhouette))

                # Preprocess the batches and yield them
                Y_batch = [np.array(Y_batch_params), np.array(Y_batch_pc)]
                X_batch = np.array(X_batch, dtype="float32")
                X_batch /= 255
                X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1))

                self.debug_X = X_batch
                self.debug_Y = Y_batch

            else:
                X_batch = self.debug_X
                Y_batch = self.debug_Y

        else:
            # Split of random and real data
            num_artificial = int(np.round(self.frac_randomised * self.batch_size))
            num_real = int(self.batch_size - num_artificial)

            # Retrieve a random batch of parameters from the data directory
            if num_real > 0:
                data = np.array(os.listdir(self.data_dir))
                Y_batch_ids = data[np.random.randint(low=0, high=data.shape[0], size=num_real)]
            else:
                Y_batch_ids = []

            Y_batch_params = []
            Y_batch_pc = []
            X_batch = []
            for Y_id in Y_batch_ids:
                # Fetch the real data
                Y = np.load(os.path.join(self.data_dir, Y_id))

                # Add a small amount of noise to the data
                Y += np.random.uniform(low=-self.noise, high=self.noise, size=Y.shape)
                Y_batch_params.append(Y)

                # Now generate the silhouette from the SMPL meshes
                # Create the body mesh
                pose = Y[:72]
                beta = Y[72:82]
                trans = Y[82:]
                pointcloud = smpl.set_params(pose.reshape((24, 3)), beta, trans)

                # Render the silhouette
                silhouette = Mesh(pointcloud=pointcloud).render_silhouette(dim=self.img_dim, show=False)
                X_batch.append(np.array(silhouette))

            for i in range(num_artificial):
                # Generate artificial data
                pose = 0.65 * (np.random.rand(smpl.pose_shape[0], smpl.pose_shape[1]) - 0.5)
                beta = 0.2 * (np.random.rand(smpl.beta_shape[0]) - 0.5)
                trans = np.zeros(smpl.trans_shape[0])
                #trans = 0.1 * (np.random.rand(smpl.trans_shape[0]) - 0.5)

                # Create the body mesh
                pointcloud = smpl.set_params(beta=beta, pose=pose, trans=trans)
                Y_batch_params.append(np.concatenate([pose.ravel(), beta, trans]))
                Y_batch_pc.append(pointcloud)

                # Render the silhouette
                silhouette = Mesh(pointcloud=pointcloud).render_silhouette(dim=self.img_dim, show=False)
                X_batch.append(np.array(silhouette))

            # Preprocess the batches and yield them
            Y_batch = [np.array(Y_batch_params), np.array(Y_batch_pc)]
            X_batch = np.array(X_batch, dtype="float32")
            X_batch /= 255
            X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1))

        #print("X_batch shape " + str(X_batch.shape))
        #print("Y_batch shape " + str(Y_batch.shape))
        #X_batch = list(X_batch)

        return X_batch, Y_batch


class ParamGenerator:
    def __init__(self, data_dir, smpl, batch_size=32, frac_randomised=0.2, noise=0.01):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.frac_randomised = frac_randomised  # fraction of parameters to generate randomly in each batch
        self.noise = noise
        self.smpl = smpl

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(os.listdir(self.data_dir)) / self.batch_size))
        pass

    def __getitem__(self, item):
        """ Yield batches of data """
        # Load the SMPL model
        #smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/./basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        smpl = self.smpl

        # @TODO: Add functionality for loading parameters
        for i in range(self.batch_size):
            # Generate artificial data
            pose = 0.65 * (np.random.rand(smpl.pose_shape[0], smpl.pose_shape[1]) - 0.5)
            beta = 0.2 * (np.random.rand(smpl.beta_shape[0]) - 0.5)
            trans = np.zeros(smpl.trans_shape[0])
            #trans = 0.1 * (np.random.rand(smpl.trans_shape[0]) - 0.5)

            # Create the body mesh
            pointcloud = smpl.set_params(beta=beta, pose=pose, trans=trans)
            X_batch.append(np.concatenate([pose.ravel(), beta, trans]))
            Y_batch.append(pointcloud)

        # Preprocess the batches and yield them
        X_batch = np.array(X_batch, dtype="float32")
        Y_batch = np.array(Y_batch, dtype="float32")

        return X_batch, Y_batch


if __name__ == "__main__":
    smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/./basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    data_gen = SilhouetteDataGenerator("../../../data/AMASS/train/", smpl, batch_size=16, img_dim=(256, 256), frac_randomised=1.0)
    X_batch, Y_batch = data_gen.__getitem__(0)

    print(X_batch.shape)
