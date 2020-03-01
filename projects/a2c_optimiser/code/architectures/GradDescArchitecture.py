import sys
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, Flatten, Conv2D, Lambda, Concatenate, Dropout, BatchNormalization, MaxPooling2D, AveragePooling2D, Embedding, Reshape, Multiply, Add
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.initializers import RandomUniform

sys.path.append('/data/cvfs/hjhb2/projects/deep_optimiser/code/keras_rotationnet_v2_demo_for_hidde/')
sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from points3d import Points3DFromSMPLParams, get_parameters
from smpl_np_rot_v6 import load_params
#from smpl_tf import smpl_model, smpl_model_batched
from render_mesh import Mesh
from architecture_helpers import custom_mod, init_emb_layers, false_loss, no_loss, cat_xent, mape, scaled_tanh, pos_scaled_tanh, scaled_sigmoid, centred_linear, get_mesh_normals, load_smpl_params, get_pc, get_sin_metric, emb_init_weights


def GradDescArchitecture(param_trainable, init_wrapper, smpl_params, input_info, faces, emb_size=1000):
    """ Gradient descent optimisation network architecture """
    # An embedding layer is required to optimise the parameters
    optlearner_input = Input(shape=(1,), name="embedding_index")

    # Initialise the embedding layers
    emb_layers = init_emb_layers(optlearner_input, emb_size, param_trainable, init_wrapper)
    print("emb layer shape: " + str(emb_layers[0].shape))
    optlearner_params = Concatenate(axis=1, name="parameter_embedding")(emb_layers)
    print("Embedding shape: " + str(optlearner_params.shape))
    #exit(1)
    optlearner_params = Reshape(target_shape=(85,), name="learned_params")(optlearner_params)
    print("optlearner parameters shape: " +str(optlearner_params.shape))
    #exit(1)

    # Ground truth parameters and point cloud are inputs to the model as well
    gt_params = Input(shape=(85,), name="gt_params")
    gt_pc = Input(shape=(6890, 3), name="gt_pc")
    print("gt parameters shape: " + str(gt_params.shape))
    print("gt point cloud shape: " + str(gt_pc.shape))

    # Compute the true offset (i.e. difference) between the ground truth and learned parameters
    #pi = K.constant(np.pi)
    delta_d = Lambda(lambda x: x[0] - x[1], name="delta_d")([gt_params, optlearner_params])
    #delta_d = Lambda(lambda x: x[0] - x[1], name="delta_d_no_mod")([gt_params, optlearner_params])
    #delta_d = Lambda(lambda x: K.tf.math.floormod(x - pi, 2*pi) - pi, name="delta_d")(delta_d)  # custom modulo 2pi of delta_d
    #delta_d = custom_mod(delta_d, pi, name="delta_d")  # custom modulo 2pi of delta_d
    print("delta_d shape: " + str(delta_d.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned parameters and the ground truth parameters
    #false_loss_delta_d = Lambda(lambda x: K.mean(K.square(x), axis=1))(delta_d)
    false_loss_delta_d = Lambda(lambda x: K.mean(K.square(x), axis=1))(delta_d)
    print("delta_d loss shape: " + str(false_loss_delta_d.shape))
    #exit(1)
    false_loss_delta_d = Reshape(target_shape=(1,), name="delta_d_mse")(false_loss_delta_d)
    print("delta_d loss shape: " + str(false_loss_delta_d.shape))

    # Load SMPL model and get necessary parameters
    optlearner_pc = get_pc(optlearner_params, smpl_params, input_info, faces)

    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_diff = Lambda(lambda x: x[0] -  x[1])([gt_pc, optlearner_pc])
    pc_euclidean_dist = Lambda(lambda x: K.sum(K.square(x),axis=-1))(pc_euclidean_diff)
    #pc_euclidean_dist = Lambda(lambda x: K.square(x))(pc_euclidean_diff)
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    #false_loss_pc = Lambda(lambda x: x, name="pc_mean_euc_dist")(pc_euclidean_dist)
    print("point cloud loss shape: " + str(false_loss_pc.shape))
    #exit(1)

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc]


