import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


# Visualisation dir
vis_dir = "/data/cvfs/hjhb2/projects/mesh_rendering/experiments/2019-12-15_09:06:35/logs/control/"

# Exclude these losses
exclusion_losses = []  #["lr_0.2"]

# Get all the loss files
losses_path = {}
for root, dirs, _ in os.walk(vis_dir):
    for subdir in dirs:
        if subdir not in exclusion_losses:
            #losses_path[subdir] = root + "/" + subdir + '/losses.txt'
            #losses_path[subdir] = root + "/" + subdir + '/mesh_losses.txt'
            losses_path[subdir] = root + "/" + subdir + '/delta_d.txt'
        #print(losses_path[subdir])

# Load the deep optimiser loss file
#losses_path["deep_opt"] = "/data/cvfs/hjhb2/projects/mesh_rendering/experiments/2019-12-15_09:06:35/logs/model.15499-561.11.hdf5" + "/losses.txt"
#losses_path["deep_opt"] = "/data/cvfs/hjhb2/projects/mesh_rendering/experiments/2019-12-15_09:06:35/logs/model.15499-561.11.hdf5" + "/mesh_losses.txt"
losses_path["deep_opt_pc"] = "/data/cvfs/hjhb2/projects/mesh_rendering/experiments/2019-12-15_09:06:35/logs/model.15499-561.11.hdf5" + "/delta_d.txt"
#losses_path["deep_opt_mesh_normals"] = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-23_10:31:21/logs/model.500-0.56.hdf5" + "/delta_d.txt"
#losses_path["deep_opt_mesh_normals"] = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-23_10:31:21/logs/model.250-inf.hdf5" + "/delta_d.txt"
losses_path["deep_opt_combined_1"] = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-28_08:31:10/logs/model.3850-0.0554.hdf5" + "/delta_d.txt"
losses_path["deep_opt_combined_2"] = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-27_18:02:01/logs/model.2100-0.0350.hdf5" + "/delta_d.txt"

# Load all the loss files
losses = {}
for subdir, loss_path in losses_path.items():
    param1 = False
    param56 = False
    param59 = False
    param1_losses = []
    param56_losses = []
    param59_losses = []
    try:
        with open(loss_path, mode='r') as f:
            for s in f.readlines():
                if param1:
                    param1_losses.append(float(s[10:19]))
                    param1 = False
                if param56:
                    param56_losses.append(float(s[10:19]))
                    param56 = False
                if param59:
                    param59_losses.append(float(s[10:19]))
                    param59 = False

                if "param_01" in s:
                    param1 = True
                if "param_56" in s:
                    param56 = True
                if "param_59" in s:
                    param59 = True

                #s = json.loads(s)
                #loss_list.append(s["pc_mean_euc_dist_loss"])
                #loss_list.append(s["loss"])
        param1_losses = [np.sin(0.5*x)**2 for x in param1_losses]
        param56_losses = [np.sin(0.5*x)**2 for x in param56_losses]
        param59_losses = [np.sin(0.5*x)**2 for x in param59_losses]

        params = np.mean(np.column_stack([param1_losses, param56_losses, param59_losses]),axis=1)
        #print(params)
        #exit()
        if len(params) >= 50:
            losses[subdir] = params[1:50]
        else:
            losses[subdir] = params
    finally:
        f.close()

#include_losses = ["lr_0.2", "lr_0.1", "lr_0.01", "lr_0.001", "lr_0.0001", "deep_opt_pc", "deep_opt_mesh_normals", "deep_opt_combined"]
include_losses = ["lr_0.1", "lr_0.01", "lr_0.001", "lr_0.0001", "deep_opt_pc", "deep_opt_combined_1", "deep_opt_combined_2"]
#include_losses = ["lr_0.2", "lr_0.1", "lr_0.01", "lr_0.001", "lr_0.0001", "deep_opt_pc"]
loss_df = pd.DataFrame(losses)
loss_df.index.name = "opt. step"
loss_df.index += 1
loss_df = loss_df[include_losses]
print(loss_df)

loss_df.plot(linewidth=3)
plt.ylabel("Loss", fontsize=16)
plt.xlabel("Optimisation step", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(prop={'size': 14})
plt.show()

