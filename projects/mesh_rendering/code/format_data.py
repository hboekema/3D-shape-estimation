import os
import glob
import numpy as np

from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    data_dir = "../../../data/AMASS/"
    data_sub_dir = "SSM_synced/"

    # Fetch all of the smpl parameters
    smpl_params = []
    for np_name in glob.glob(os.path.join(data_dir, data_sub_dir, '*.np[yz]')):
        data = np.load(np_name)

        # Get the poses
        poses = data["poses"]
        poses = poses.reshape(poses.shape[0], 52, 3)[:, :24, :]
        # Get the betas
        betas = data["betas"][:10]
        # Get the trans
        trans = data["trans"]

        # Concatenate the data to fit in a single array
        for i in range(poses.shape[0]):
            smpl_params.append(np.concatenate([poses[i].ravel(), betas, trans[i]]))

    # Randomly draw data for the test and validation steps
    X_train, X_val = train_test_split(smpl_params, test_size=0.1)
    X_train, X_test = train_test_split(X_train, test_size=0.1)

    # Store the parameters in each of the arrays in the appropriate directories
    for i, x in enumerate(X_train):
        np.save(os.path.join(data_dir, "train/train_sample_{:04d}.npy".format(i)), x)

    for i, x in enumerate(X_val):
        np.save(os.path.join(data_dir, "val/val_sample_{:04d}.npy".format(i)), x)

    for i, x in enumerate(X_test):
        np.save(os.path.join(data_dir, "test/test_sample_{:04d}.npy".format(i)), x)


