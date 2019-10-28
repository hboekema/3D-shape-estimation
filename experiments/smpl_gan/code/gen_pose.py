""" Generate arbitrary pose parameters using real data """

import os

import glob
import numpy as np
from sklearn.model_selection import train_test_split

from gan_network import GAN

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Load the data
data_dir = "../../../data/AMASS/SSM_synced/"
X_poses = []
X_betas = []
X_trans = []

# Load all files in the data directory
for np_name in glob.glob(os.path.join(data_dir, '*.np[yz]')):
    data = np.load(np_name)

    # Get the poses
    poses = data["poses"]
    poses = poses.reshape((poses.shape[0], 52, 3))[:, :24, :]
    poses[:, 0] = [0, 0, 0]     # make it easier to train the model by trimming the global orientation
    X_poses.append(poses)

    # Get the betas
    betas = data["betas"][:10]
    X_betas.append(betas)

    # Get the trans
    trans = data["trans"]
    X_trans.append(trans)

X_poses = np.concatenate(X_poses)
X_betas = np.concatenate(X_betas)
X_trans = np.concatenate(X_trans)

# Split the poses dataset into train and validation set
X_train, X_val = train_test_split(X_poses, test_size=0.1)

# Now create and train the GAN on the pose data
target_dim = (24, 4)

print("Training network...")
gan = GAN(target_dim)
gan.fit(X_train, X_val)
gan.train(epochs=100, steps_per_epoch=10, batch_size=64, epoch_save_period=10)

# Now create a few fake poses to test the model
# noise = np.random.randn(10, 100)
# fake_poses = gan.generator.predict(noise)
#
# for i, pose in enumerate(fake_poses):
#     np.save("../poses/fake_pose.{:03d}.npy".format(i), pose)
