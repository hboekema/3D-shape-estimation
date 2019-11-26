import os
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
#import cv2

from smpl_np import SMPLModel
from render_mesh import Mesh


def create_data(smpl, num=1000, data_dir=None, img_dim=(256, 256)):
    """ Create artificial parameters and their resulting silhouettes and store them in the specified directory """
    X_data = []
    Y_data_params = []
    Y_data_pc = []
    print("Generating data...")
    for i in tqdm(range(num)):
        # Generate artificial data
        pose = 0.65 * (np.random.rand(smpl.pose_shape[0], smpl.pose_shape[1]) - 0.5)
        beta = 0.2 * (np.random.rand(smpl.beta_shape[0]) - 0.5)
        trans = np.zeros(smpl.trans_shape[0])
        #trans = 0.1 * (np.random.rand(smpl.trans_shape[0]) - 0.5)

        # Create the body mesh
        pointcloud = smpl.set_params(beta=beta, pose=pose, trans=trans)
        Y_data_params.append(np.concatenate([pose.ravel(), beta, trans]))
        Y_data_pc.append(pointcloud)

        # Render the silhouette
        silhouette = Mesh(pointcloud=pointcloud).render_silhouette(dim=img_dim, show=False)
        X_data.append(np.array(silhouette))

    # Preprocess the data
    Y_data = [np.array(Y_data_params), np.array(Y_data_pc)]
    X_data = np.array(X_data, dtype="float32")
    X_data /= 255
    X_data = X_data.reshape((num, img_dim[0], img_dim[1], 1))

    if data_dir is not None:
        print("Saving data...")
        for i in tqdm(range(num)):
            # Save the data
            data_obj = {"silhouette": X_data[i], "pointcloud": Y_data_pc[i], "parameters": Y_data_params[i]}
            with open(os.path.join(data_dir, "sample_{:03d}.obj".format(i+1)), "wb") as handle:
                pickle.dump(data_obj, handle)

    return X_data, Y_data

if __name__ == "__main__":
    smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    num_samples = 1000
    silhouettes, Y_data = create_data(smpl, num=num_samples, data_dir="../data/test/", img_dim=(256, 256))
#    params = Y_data[0]
#    pcs = Y_data[1]
#
#    index = np.random.randint(low=1, high=num_samples)
#    f = open("../data/val/sample_{:03d}.obj".format(index+1), 'r')
#    loaded_data = pickle.load(f)
#    loaded_silh = loaded_data["silhouette"]
#    loaded_pc = loaded_data["pointcloud"]
#    loaded_params = loaded_data["parameters"]
#
#    silh_reshaped = loaded_silh.reshape((256, 256))
#    silh_reshaped *= 255
#    print("loaded silh (reshaped) shape: " + str(silh_reshaped.shape))
#    print("loaded params shape: " + str(loaded_params.shape))
#    print("loaded pc shape: " + str(loaded_pc.shape))
#    plt.imshow(silh_reshaped.astype("uint8"), cmap="gray")
#    plt.show()


    #assert np.all([silhouettes[index], loaded_silh])
    #assert np.all([pcs[index], loaded_pc])
    #assert np.all([params[index], loaded_params])

