""" Generate artificial data """

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from smpl_np import SMPLModel
sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from smpl_np_rot_v6 import print_mesh, print_point_clouds
from render_mesh import Mesh
from tqdm import tqdm
from training_helpers import format_distractor_dict, offset_params
#import cv2


def gen_data(POSE_OFFSET, PARAMS_TO_OFFSET, smpl, data_samples=10000, save_dir=None, render_silhouette=True):
    """ Generate random body poses """
    POSE_OFFSET = format_distractor_dict(POSE_OFFSET, PARAMS_TO_OFFSET)

    zero_params = np.zeros(shape=(85,))
    zero_pc = smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85])
    #print("zero_pc: " + str(zero_pc))

    # Generate and format the data
    X_indices = np.array([i for i in range(data_samples)])
    X_params = np.array([zero_params for i in range(data_samples)], dtype="float32")
    if not all(value == 0.0 for value in POSE_OFFSET.values()):
        X_params = offset_params(X_params, PARAMS_TO_OFFSET, POSE_OFFSET)
        X_pcs = np.array([smpl.set_params(beta=params[72:82], pose=params[0:72], trans=params[82:85]) for params in X_params])
    else:
        X_pcs = np.array([zero_pc for i in range(data_samples)], dtype="float32")

    if render_silhouette:
        X_silh = []
        print("Generating silhouettes...")
        for pc in tqdm(X_pcs):
            # Render the silhouette from the point cloud
            silh = Mesh(pointcloud=pc).render_silhouette(show=False)
            X_silh.append(silh)

        X_silh = np.array(X_silh)
        print("Finished generating data.")

    if save_dir is not None:
        # Save the generated data in the given location
        print("Saving generated samples...")
        for i in tqdm(range(data_samples)):
            sample_id = "sample_{:05d}".format(i+1)
            if render_silhouette:
                np.savez(save_dir + sample_id + ".npz", smpl_params=X_params[i], pointcloud=X_pcs[i], silhouette=X_silh[i])
            else:
                np.savez(save_dir + sample_id + ".npz", smpl_params=X_params[i], pointcloud=X_pcs[i], silhouette=X_silh[i])

        print("Finished saving.")

    if render_silhouette:
        return X_params, X_pcs, X_silh
    else:
        return X_params, X_pcs


def load_data(load_dir, num_samples=10000, load_silhouettes=False):
    """ Load previously generated data """
#    param_dir = load_dir + "smpl_params/"
#    pc_dir = load_dir + "pointclouds/"
#    silh_dir = load_dir + "silhouettes/"

    X_params = []
    X_pcs = []
    X_silh = []
    print("Loading data from '{}'...".format(load_dir))
    for i in tqdm(range(num_samples)):
        sample_id = "sample_{:05d}".format(i+1)

#        params = np.loadtxt(param_dir + sample_id + ".csv")
#        pc = Mesh(filepath=pc_dir + sample_id + ".obj").verts
#        if load_silhouettes:
#            silh = cv2.imread(silh_dir + sample_id + ".png", cv2.IMREAD_GRAYSCALE)
#            X_silh.append(silh)

        X_data = np.load(load_dir + sample_id + ".npz")

        X_params.append(X_data['smpl_params'])
        X_pcs.append(X_data['pointcloud'])
        X_silh.append(X_data['silhouette'])

    X_params = np.array(X_params)
    X_pcs = np.array(X_pcs)
    X_silh = np.array(X_silh)

    print("Finished.")
    if load_silhouettes:
        return X_params, X_pcs, X_silh
    else:
        return X_params, X_pcs


if __name__ == "__main__":
    np.random.seed(10)
    #np.random.seed(12)
    np.random.seed(11)

    #data_samples = 100
    #data_samples = 10000
    data_samples = 20000

    #POSE_OFFSET =  0.0
    POSE_OFFSET = {"param_01": "pi", "other": 0.3}
    PARAMS_TO_OFFSET = "all_pose_and_global_rotation"
    #PARAMS_TO_OFFSET = ["param_14", "param_17", "param_59", "param_56"]
    #PARAMS_TO_OFFSET = ["param_01", "param_59", "param_56"]
    #PARAMS_TO_OFFSET = ["param_01", "param_59"]
    #PARAMS_TO_OFFSET = ["param_59", "param_56"]
    #PARAMS_TO_OFFSET = ["param_01"]
    #PARAMS_TO_OFFSET = ["param_56"]
    #PARAMS_TO_OFFSET = ["param_59"]

    param_ids = ["param_{:02d}".format(i) for i in range(85)]
    if PARAMS_TO_OFFSET == "all_pose_and_global_rotation":
        not_trainable = [0, 2]
        #not_trainable = []
        #trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
        trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 66]
        PARAMS_TO_OFFSET = [param_ids[index] for index in trainable_params_indices]

    # Generate the data from the SMPL parameters
    smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    root_data_dir = "/data/cvfs/hjhb2/data/artificial/"
    os.system('mkdir ' + root_data_dir)
    offsets = "full_y_rot_and_small_pose/"
    os.system('mkdir ' + root_data_dir + offsets)
    #data_type = "train/"
    #data_type = "val/"
    data_type = "test/"

    save_dir = root_data_dir + offsets + data_type
    os.system('mkdir ' + save_dir)

    gen_data(POSE_OFFSET, PARAMS_TO_OFFSET, smpl, data_samples, save_dir)
    #X_params, X_pcs, X_silh = load_data(save_dir, data_samples, load_silhouettes=True)

    #Mesh(pointcloud=X_pcs[0]).render_silhouette()

