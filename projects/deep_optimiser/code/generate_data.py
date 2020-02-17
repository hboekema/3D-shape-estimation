""" Generate artificial data """

import sys
import os
import numpy as np

from smpl_np import SMPLModel
sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from smpl_np_rot_v6 import print_mesh, print_point_clouds
from render_mesh import Mesh
from tqdm import tqdm
import cv2


def gen_data(param_trainable, smpl, data_samples=10000, save_dir=None):
    """ Generate random body poses """
    X_params = 0.2 * 2*(np.random.rand(data_samples, 85) - 0.5)

    k = np.pi
    for param, trainable in param_trainable.items():
        if trainable:
            param_int = int(param[6:8])
            X_params[:, param_int] = 2 * k * (np.random.rand(data_samples) - 0.5)

    X_pcs = []
    X_silh = []
    for params in X_params:
        # Render the point cloud
        pc = smpl.set_params(beta=params[72:82], pose=params[0:72], trans=params[82:85])
        X_pcs.append(pc)

        # Now render the silhouette from this
        silh = Mesh(pointcloud=pc).render_silhouette(show=False)
        X_silh.append(silh)

    X_pcs = np.array(X_pcs)
    X_silh = np.array(X_silh)

    if save_dir is not None:
        # Save the generated data in the given location
        param_dir = save_dir + "smpl_params/"
        pc_dir = save_dir + "pointclouds/"
        silh_dir = save_dir + "silhouettes/"
        os.system('mkdir ' + param_dir)
        os.system('mkdir ' + pc_dir)
        os.system('mkdir ' + silh_dir)
        faces = smpl.faces

        for i in range(data_samples):
            sample_id = "sample_{:05d}".format(i+1)
            params = X_params[i]
            pc = X_pcs[i]
            silh = X_silh[i]

            np.savetxt(param_dir + sample_id + ".csv", params, delimiter=",")
            print_mesh(pc_dir + sample_id + ".obj", pc, faces)
            cv2.imwrite(silh_dir + sample_id + ".png", silh.astype("uint8"))

    return X_params, X_pcs, X_silh


def load_data(load_dir, num_samples=10000, load_silhouettes=False):
    """ Load previously generated data """
    param_dir = load_dir + "smpl_params/"
    pc_dir = load_dir + "pointclouds/"
    silh_dir = load_dir + "silhouettes/"

    X_params = []
    X_pcs = []
    X_silh = []
    print("Loading data from '{}'...".format(load_dir))
    for i in tqdm(range(num_samples)):
        sample_id = "sample_{:05d}".format(i+1)

        params = np.loadtxt(param_dir + sample_id + ".csv")
        pc = Mesh(filepath=pc_dir + sample_id + ".obj").verts
        if load_silhouettes:
            silh = cv2.imread(silh_dir + sample_id + ".png", cv2.IMREAD_GRAYSCALE)
            X_silh.append(silh)

        X_params.append(params)
        X_pcs.append(pc)

    X_params = np.array(X_params)
    X_pcs = np.array(X_pcs)
    X_silh = np.array(X_silh)

    print("Finished.")
    if load_silhouettes:
        return X_params, X_pcs, X_silh
    else:
        return X_params, X_pcs


if __name__ == "__main__":
    #np.random.seed(10)
    np.random.seed(11)
    #np.random.seed(12)

    data_samples = 10000

    param_ids = ["param_{:02d}".format(i) for i in range(85)]
    #not_trainable = [0, 1, 2]
    #trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 70]
    #trainable_params = [param_ids[index] for index in trainable_params_indices]

    #trainable_params = ["param_00", "param_01", "param_02", "param_56", "param_57", "param_58", "param_59", "param_60", "param_61"]
    trainable_params = ["param_01", "param_59", "param_56"]
    #trainable_params = ["param_59", "param_56"]
    param_trainable = { param: (param in trainable_params) for param in param_ids }
    # Generate the data from the SMPL parameters
    smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    #save_dir = "/data/cvfs/hjhb2/data/artificial/train/"
    save_dir = "/data/cvfs/hjhb2/data/artificial/test/"
    #save_dir = "/data/cvfs/hjhb2/data/artificial/val/"
    gen_data(param_trainable, smpl, data_samples, save_dir)
    #X_params, X_pcs, X_silh = load_data(save_dir, data_samples)

    #Mesh(pointcloud=X_pcs[0]).render_silhouette()

