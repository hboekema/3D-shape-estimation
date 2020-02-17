import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import cv2
from keras.models import Model
from keras.optimizers import Adam, SGD
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import copy

from callbacks import OptLearnerPredOnEpochEnd
from silhouette_generator import OptLearnerDataGenerator, OptLearnerExtraOutputDataGenerator
#from optlearner import OptLearnerStaticArchitecture, OptLearnerStaticSinArchitecture, OptLearnerBothNormalsStaticSinArchitecture,OptLearnerMeshNormalStaticArchitecture, OptLearnerMeshNormalStaticModArchitecture, OptLearnerCombinedStaticModArchitecture, OptLearnerMeshNormalStaticSinArchitecture, OptLearnerArchitecture, OptLearnerExtraOutputArchitecture, no_loss, false_loss, load_smpl_params, emb_init_weights
from architectures.OptLearnerCombinedStaticModArchitecture import OptLearnerCombinedStaticModArchitecture
from architectures.OptLearnerMeshNormalStaticArchitecture import OptLearnerMeshNormalStaticArchitecture
from architectures.OptLearnerMeshNormalStaticModArchitecture import OptLearnerMeshNormalStaticModArchitecture
from architectures.FullOptLearnerStaticArchitecture import FullOptLearnerStaticArchitecture
from architectures.Conv1DFullOptLearnerStaticArchitecture import Conv1DFullOptLearnerStaticArchitecture
from architectures.architecture_helpers import false_loss, no_loss, load_smpl_params, emb_init_weights
from smpl_np import SMPLModel
from render_mesh import Mesh
from euler_rodrigues_transform import rodrigues_to_euler

parser = ArgumentParser()
#parser.add_argument("--model", help="Path to the model weights")
parser.add_argument("--data", help="Path to the data directory")
args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
print("GPU used: |" + str(os.environ["CUDA_VISIBLE_DEVICES"]) + "|")

if args.data is None:
    args.data = "../data/train/"

#np.random.seed(10)
np.random.seed(11)

# Experiment directory
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-26_13:07:22/"
#model_name = "model.200-0.08.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-27_10:53:20/"
#model_name = "model.700-0.0360.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-27_10:51:19/"
#model_name = "model.700-0.0361.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-27_11:26:52/"
#model_name = "model.700-0.0377.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-27_16:15:10/"
#model_name = "model.100-0.0379.hdf5"
#model_name = "model.2350-0.0369.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-27_18:02:01/"
#model_name = "model.2100-0.0350.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-28_08:31:10/"
#model_name = "model.450-0.0571.hdf5"
#model_name = "model.300-0.0570.hdf5"
#model_name = "model.1200-0.0564.hdf5"
#model_name = "model.3850-0.0554.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-28_10:34:25/"
#model_name = "model.100-0.9813.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-27_18:23:07/"
#model_name = "model.2150-0.0566.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-28_12:08:29/"
#model_name = "model.50-0.0576.hdf5"
#model_name = "model.100-0.0578.hdf5"
#model_name = "model.600-0.0570.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-28_17:07:27/"
#model_name = "model.50-0.0582.hdf5"
#model_name = "model.100-0.0584.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-28_18:31:15"
#model_name = "model.200-0.0127.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-28_17:07:27/"
#model_name = "model.2500-0.0548.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-29_22:25:39/"
#model_name = "model.500-0.0288.hdf5"
#model_name = "model.800-0.0288.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-29_22:26:26/"
#model_name = "model.500-0.0133.hdf5"
#model_name = "model.800-0.0136.hdf5"
#model_name = "model.950-0.0134.hdf5"

# Single-parameter experiments
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-30_14:15:37/"
#model_name = "model.250-0.0132.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-30_14:12:44/"
#model_name = "model.250-0.0064.hdf5"
#model_name = "model.1100-0.0057.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-30_14:13:59/"
#model_name = "model.200-0.0074.hdf5"
#model_name = "model.1150-0.0087.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-30_14:14:48/"
#model_name = "model.250-0.0093.hdf5"
#model_name = "model.1100-0.0105.hdf5"

# 32-batch size experiments
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-30_18:31:20/"
#model_name = "model.2400-0.0141.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-30_21:05:05/"
#model_name = "model.1600-0.0130.hdf5"

# PC and mesh normals experiments
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-31_11:50:24/"
#model_name = "model.100-0.0707.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-31_11:53:24/"
#model_name = "model.100-0.0063.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-31_11:57:37/"
#model_name = "model.100-0.0057.hdf5"
#model_name = "model.150-0.0067.hdf5"

# Unmodded MSE experiment
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-01-31_22:45:14/"
#model_name = "model.700-0.0068.hdf5"

# Basic experiments
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-01_15:13:57/"
#model_name = "model.100-0.0780.hdf5"
#model_name = "model.1300-0.0211.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-02_08:42:52/"
#model_name = "model.100-0.0235.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-02_08:34:29/"
#model_name = "model.100-0.0242.hdf5"

# Basic experiments on zero pose
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-02_14:02:24/"
#model_name = "model.1550-0.0095.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-02_14:02:59/"
#model_name = "model.1550-0.0089.hdf5"
# 3d points
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/3d_points_2020-02-13_10:34:28/"
#model_name = "model.20-0.0051.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/3d_points_2020-02-13_10:35:15/"
#model_name = "model.20-0.0052.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/3d_points_2020-02-13_10:33:41/"
#model_name = "model.20-0.0066.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/3d_points_2020-02-13_10:31:18/"
#model_name = "model.10-0.0218.hdf5"
# normals
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-13_10:39:04/"
#model_name = "model.20-0.0043.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-13_10:40:10/"
#model_name = "model.30-0.0045.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-13_10:38:14/"
#model_name = "model.20-0.0071.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-13_10:37:29/"
#model_name = "model.30-0.0191.hdf5"
# random network
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/random_update_2020-02-13_11:30:28/"
#model_name = "model.01-1.0088.hdf5"

# Euler angle models
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-03_08:51:08"
#model_name = "model.100-0.0099.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-03_08:56:46/"
#model_name = "model.150-0.0084.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-07_11:34:28/"
#model_name = "model.50-0.0053.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-08_13:39:26/"
#model_name = "model.200-0.0081.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-08_13:46:55/"
#model_name = "model.1000-0.0084.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-09_10:09:27/"
#model_name = "model.50-0.0491.hdf5"
#model_name = "model.100-0.0480.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-09_10:08:13/"
#model_name = "model.50-0.0351.hdf5"
#model_name = "model.100-0.0343.hdf5"

# Distractor experiments
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-11_13:50:22/"
#model_name = "model.150-0.0204.hdf5"

# Full vertex selection models
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-09_12:58:26/"
#model_name = "model.3900-0.0409.hdf5"

# Convolutional models
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/mesh_normal_optimiser_2020-02-11_15:53:15/"
#model_name = "model.100-0.0050.hdf5"
#model_name = "model.1300-0.0043.hdf5"
#model_name = "model.3450-0.0060.hdf5"
exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/grad_desc_2020-02-13_13:19:45/"
#model_name = "model.100-0.0018.hdf5"
#model_name = "model.200-0.0018.hdf5"
#model_name = "model.350-0.0019.hdf5"
model_name = "model.900-0.0020.hdf5"
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/3d_points_2020-02-13_17:49:23/"
#model_name = "model.500-0.0112.hdf5"

# Elbows and knees
#exp_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/experiments/grad_desc_2020-02-13_15:07:38/"
#model_name = "model.750-0.0216.hdf5"


model = exp_dir + "models/" + model_name
logs_dir = exp_dir + "logs/" + model_name + "/"
#logs_dir = exp_dir + "logs/" + model_name + "_small_delta/"
os.system('mkdir ' + logs_dir)
control_logs_dir = exp_dir + "logs/control/"
#control_logs_dir = exp_dir + "logs/control_small_delta/"
os.system('mkdir ' + control_logs_dir)
test_vis_dir = exp_dir + "test_vis/" + model_name + "/"
#test_vis_dir = exp_dir + "test_vis/" + model_name + "_small_delta/"
os.system('mkdir ' + test_vis_dir)
control_dir = exp_dir + "test_vis/" + "control/"
#control_dir = exp_dir + "test_vis/" + "control_small_delta/"
os.system('mkdir ' + control_dir)


# Generate the data from the SMPL parameters
param_ids = ["param_{:02d}".format(i) for i in range(85)]
#not_trainable = [0, 2]
#trainable_params_indices = [index for index in range(85) if index not in not_trainable and index < 72]
#trainable_params = [param_ids[index] for index in trainable_params_indices]
#trainable_params = ["param_00", "param_01", "param_02", "param_56", "param_57", "param_58", "param_59", "param_60", "param_61"]
#trainable_params = ["param_01", "param_56", "param_59"]
#trainable_params = ["param_14", "param_17", "param_56", "param_59"]
#trainable_params = ["param_01", "param_59"]
#trainable_params = ["param_01", "param_56"]
#trainable_params = ["param_01"]
#trainable_params = ["param_56"]
trainable_params = ["param_59"]
param_trainable = { param: (param in trainable_params) for param in param_ids }


smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
zero_params = np.zeros(shape=(85,))
zero_pc = smpl.set_params(beta=zero_params[72:82], pose=zero_params[0:72].reshape((24,3)), trans=zero_params[82:85])
#print("zero_pc: " + str(zero_pc))
#base_params = 0.2 * (np.random.rand(85) - 0.5)
#base_pc = smpl.set_params(beta=base_params[72:82], pose=base_params[0:72].reshape((24,3)), trans=base_params[82:85])

# Parameters
MODE = "RODRIGUES"
#MODE = "EULER"
DISTRACTOR = np.pi
#DISTRACTOR = 0.3
DATA_DIR = "/data/cvfs/hjhb2/data/artificial/"

data_samples = 10000
#data_samples = 1000
#data_samples = 10

X_indices = np.array([i for i in range(data_samples)])
#X_params = 0.2 * np.random.rand(data_samples, 85)
X_params = np.array([zero_params for i in range(data_samples)], dtype="float64")
X_pcs = np.array([zero_pc for i in range(data_samples)], dtype="float64")
#X_params = np.array([base_params for i in range(data_samples)], dtype="float32")
#X_pcs = np.array([base_pc for i in range(data_samples)], dtype="float32")

def trainable_param_dist(X_params, trainable_params, k=np.pi):
    trainable_params_int = [int(param[6:8]) for param in trainable_params]
    X_params[:, trainable_params_int] = 2 * k * (np.random.rand(data_samples, len(trainable_params_int)) - 0.5)

    return X_params

#X_params = trainable_param_dist(X_params, trainable_params, DISTRACTOR)

#X_params = 0.2 * np.random.rand(data_samples, 85)
#X_pcs = []
#for params in X_params:
#    pc = smpl.set_params(beta=params[72:82], pose=params[0:72], trans=params[82:85])
#    X_pcs.append(pc)
#X_pcs = np.array(X_pcs)

if MODE == "EULER":
    # Convert from Rodrigues to Euler angles
    X_params = rodrigues_to_euler(X_params, smpl)
    #pose_rodrigues = np.array([X_params[:, i:i+3] for i in range(0, 72, 3)])
    #print(pose_rodrigues[0][0])
    #pose_rodrigues = pose_rodrigues.reshape((24, data_samples, 1, 3))
    #print(pose_rodrigues[0][0])
    #print("pose_rodrigues shape: " + str(pose_rodrigues.shape))
    #R = np.array([smpl.rodrigues(vector) for vector in pose_rodrigues])
    #print(R[0][0])
    #R = R.reshape((data_samples, 24, 3, 3))
    #pose_params = np.array([[rotation_matrix_to_euler_angles(rot_mat) for rot_mat in param_rot_mats] for param_rot_mats in R])
    #pose_params = pose_params.reshape((data_samples, 72))
    #print("pose_params shape: " + str(pose_params.shape))
    #print("other params shape: " + str(X_params[:, 72:85].shape))
    #X_params = np.concatenate([pose_params, X_params[:, 72:85]], axis=1)
    #print("X_params shape: " + str(X_params.shape))

X_data = [np.array(X_indices), np.array(X_params), np.array(X_pcs)]
#Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85))]
#Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85))]
#Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 6))]
Y_data = [np.zeros((data_samples, 85)), np.zeros((data_samples,)), np.zeros((data_samples, 6890, 3)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples,)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 85)), np.zeros((data_samples, 7))]

x_test = X_data
y_test = Y_data

# Render silhouettes for the callback data
num_samples = 5
cb_indices = X_indices[:num_samples]
cb_params = X_params[:num_samples]
cb_pcs = X_pcs[:num_samples]
X_cb = [np.array(cb_indices), np.array(cb_params), np.array(cb_pcs)]
silh_cb = []
for pc in cb_pcs:
    silh = Mesh(pointcloud=pc).render_silhouette(show=False)
    silh_cb.append(silh)

# Initialise the embedding layer
def emb_init_weights_np(emb_params):
    def emb_init_wrapper(param, offset=False):
        def emb_init(shape, dtype="float32"):
            """ Initializer for the embedding layer """
            emb_params_ = emb_params[:, param]

            if offset:
                k = np.pi
                #k = 1.0
                offset_ = k * 2 * (np.random.rand(shape[0]) - 0.5)
                emb_params_[:] += offset_

            init = np.array(emb_params_, dtype=dtype).reshape(shape)
            #print("init shape: " + str(init.shape))
            #print("init values: " + str(init))
            #exit(1)
            return init
        return emb_init
    return emb_init_wrapper

emb_initialiser = emb_init_weights_np(X_params)

# Load model
smpl_params, input_info, faces = load_smpl_params()
#optlearner_inputs, optlearner_outputs = OptLearnerStaticModArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, emb_size=data_samples)
#optlearner_inputs, optlearner_outputs = OptLearnerStaticModBinnedArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, emb_size=data_samples)
#optlearner_inputs, optlearner_outputs = OptLearnerStaticSinArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, emb_size=data_samples)
#optlearner_inputs, optlearner_outputs = OptLearnerMeshNormalStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, emb_size=data_samples)
#optlearner_inputs, optlearner_outputs = OptLearnerCombinedStaticModArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples)
#optlearner_inputs, optlearner_outputs = OptLearnerMeshNormalStaticModArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples)
#optlearner_inputs, optlearner_outputs = FullOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples)
optlearner_inputs, optlearner_outputs = Conv1DFullOptLearnerStaticArchitecture(param_trainable=param_trainable, init_wrapper=emb_initialiser, smpl_params=smpl_params, input_info=input_info, faces=faces, emb_size=data_samples)
#input_indices = [0, 2]
#output_indices = [0, 2, 3, 5]
#optlearner_model = Model(inputs=[input_ for i, input_ in enumerate(optlearner_inputs) if i in input_indices], outputs=[output for i, output in enumerate(optlearner_outputs) if i in output_indices])
optlearner_model = Model(inputs=optlearner_inputs, outputs=optlearner_outputs)
optlearner_model.load_weights(model)

# Freeze all layers except for the required embedding layers
trainable_layers_names = [param_layer for param_layer, trainable in param_trainable.items() if trainable]
trainable_layers = {optlearner_model.get_layer(layer_name): int(layer_name[6:8]) for layer_name in trainable_layers_names}
for layer in optlearner_model.layers:
    if layer.name not in trainable_layers_names:
        layer.trainable = False


# Set the weights of the embedding layer
for layer_name, trainable in param_trainable.items():
    emb_layer = optlearner_model.get_layer(layer_name)
    emb_init_ = emb_initialiser(param=int(layer_name[6:8]), offset=trainable)
    emb_layer.set_weights(emb_init_(shape=(data_samples, 1)).reshape(1, data_samples, 1))
    #print(np.array(emb_layer.get_weights()).shape)

# Compile the model
learning_rate = 0.01
optimizer = Adam(lr=learning_rate, decay=0.0)
#optimizer = SGD(learning_rate, momentum=0.0, nesterov=False)
#optlearner_model.compile(optimizer=optimizer, loss=[no_loss, false_loss, no_loss, false_loss,
#                                                    false_loss,  # pc loss
#                                                    false_loss,  # loss that updates smpl parameters
#                                                    no_loss, no_loss, no_loss],
#                                            loss_weights=[0.0, 0.0, 0.0,
#                                                        1.0*1.0, # point cloud loss
#                                                        0.0*10, # delta_d_hat loss
#                                                        0.0, # this is the loss which updates smpl parameter inputs with predicted gradient
#                                                        0.0, 0.0, 0.0])
optlearner_model.compile(optimizer=optimizer, loss=[no_loss,
                                                   false_loss, # delta_d loss (L_smpl loss)
                                                   no_loss,
                                                   false_loss, # point cloud loss (L_xent)
                                                   no_loss, # delta_d hat loss (L_delta_smpl)
                                                   no_loss, # delta_d_hat sin metric
						   no_loss, # this is the loss which updates smpl parameter inputs with predicted gradient
						   no_loss, no_loss,
                                                   no_loss # difference angle loss (L_xent)
                                                   #no_loss, no_loss, no_loss
                                                   ],
                                            loss_weights=[0.0,
                                                        0.0*1, # delta_d loss (L_smpl loss)
                                                        0.0,
                                                        0.0*1, # point cloud loss (L_xent)
                                                        0.0*1, # delta_d_hat loss (L_delta_smpl)
                                                        0.0, # delta_d_hat sin metric - always set to 0
                                                        0.0, # this is the loss which updates smpl parameter inputs with predicted gradient
                                                        0.0, 0.0,
                                                        0.0*1, # difference angle loss (L_xent)
                                                        #0.0, 0.0, 0.0
                                                        ],
                                            metrics={"delta_d_hat_sin_output": false_loss}
                                            #options=run_options,
                                            #run_metadata=run_metadata
                                            )

# Print model summary
optlearner_model.summary()


# Visualisation callback
epoch_pred_cb = OptLearnerPredOnEpochEnd(logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=test_vis_dir, period=1, trainable_params=trainable_params, visualise=False, testing=True)
epoch_pred_cb_control = OptLearnerPredOnEpochEnd(control_logs_dir, smpl, train_inputs=X_cb, train_silh=silh_cb, pred_path=control_dir, period=1, trainable_params=trainable_params,visualise=False, testing=True)


def learned_optimizer(optlearner_model, epochs=50, lr=0.1, mode="RODRIGUES"):
    metrics_names = optlearner_model.metrics_names
    print("metrics_names: " + str(metrics_names))
    named_scores = {}
    output_names = [output.op.name.split("/")[0] for output in optlearner_model.outputs]
    print("output_names: " + str(output_names))
    pred_index = output_names.index("delta_d_hat")
    #print("prediction index: " + str(pred_index))

    epoch_pred_cb.set_model(optlearner_model)
    #layer_names = ["diff_cross_product", "diff_angle_mse"]
    #intermediate_layer_model = Model(inputs=optlearner_model.input, outputs=[optlearner_model.get_layer(layer_name).output for layer_name in layer_names])

    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1))
        print("----------------------")
        epoch_pred_cb.on_epoch_begin(epoch=int(epoch), logs=named_scores)
        #print('emb layer weights'+str(emb_weights))
	#print('shape '+str(emb_weights[0].shape))
	y_pred = optlearner_model.predict(x_test)
        delta_d_hat = y_pred[pred_index]

        # Evaluate the mesh normals
        #intermediate_output = intermediate_layer_model.predict(x_test)
        #print("mesh_normals: " + str(intermediate_output[0][:5]))
        #print("mesh_normals_mse: " + str(intermediate_output[1][:5]))

        for emb_layer, param_num in trainable_layers.items():
	    emb_weights = emb_layer.get_weights()
            #emb_weights += lr * np.array(y_pred[7][:, param_num]).reshape((data_samples, 1))
            #emb_weights += lr * np.array(y_pred[8][:, param_num]).reshape((data_samples, 1))
            emb_weights += lr * np.array(delta_d_hat[:, param_num]).reshape((data_samples, 1))
	    emb_layer.set_weights(emb_weights)

        # Evaluate model performance
        #scores = optlearner_model.evaluate(x_test, y_test, batch_size=100)
        #for i, score in enumerate(scores):
        #    named_scores[metrics_names[i]] = score
        #print("scores: " + str(scores))
        #exit(1)
        epoch_pred_cb.on_epoch_end(epoch=int(epoch), logs=named_scores)
    epoch_pred_cb.set_model(optlearner_model)


def regular_optimizer(optlearner_model, epochs=50):
    # Train the model
    optlearner_model.fit(
                x_test,
                y_test,
                batch_size=32,
                epochs=epochs,
                callbacks=[epoch_pred_cb_control]
            )


if __name__ == "__main__":
    learned_optimizer(optlearner_model, lr=0.5)
    #regular_optimizer(optlearner_model, epochs=50)
    exit(1)


# Evaluate the model performance
eval_log = optlearner_model.evaluate_generator(test_gen)

eval_string = ""
for i in range(len(eval_log)):
    eval_string += str(optlearner_model.metrics_names[i]) + ": " + str(eval_log[i]) + "  "
print(eval_string)

# Predict the parameters from the silhouette and generate a mesh
X_test, Y_test = test_gen.__getitem__(0)
prediction = optlearner_model.predict(X_test)
for pred in prediction:
    print(pred.shape)

pred_params = prediction[0]
real_params = Y_test[0]

for pred in pred_params:
    pointcloud = smpl.set_params(pred[10:82], pred[0:10], pred[82:85])
    pred_silhouette = Mesh(pointcloud=pointcloud).render_silhouette()

    pointcloud = smpl.set_params(real_params[10:82], real_params[0:10], real_params[82:85])
    real_silhouette = Mesh(pointcloud=pointcloud).render_silhouette()

    diff_silhouette = abs(real_silhouette - pred_silhouette)

    comp_silh = np.concatenate([real_silhouette, pred_silhouette, diff_silhouette])
    plt.imshow(comp_silh, cmap='gray')
    plt.show()



