""" Generate artificial data """

import numpy as np
import sys
import os
import matplotlib
matplotlib.use("Agg")
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from smpl_np import SMPLModel
sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from smpl_np_rot_v6 import print_mesh, print_point_clouds
from data_helpers import gen_data

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

