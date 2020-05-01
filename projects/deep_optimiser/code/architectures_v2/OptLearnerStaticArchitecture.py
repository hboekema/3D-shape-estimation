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

sys.path.append('/data/cvfs/hjhb2/projects/mesh_rendering/code/keras_rotationnet_v2_demo_for_hidde/')
sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from points3d import Points3DFromSMPLParams, get_parameters
from smpl_np_rot_v6 import load_params
#from smpl_tf import smpl_model, smpl_model_batched
from render_mesh import Mesh
from architecture_helpers import custom_mod, init_emb_layers, false_loss, no_loss, cat_xent, mape, scaled_tanh, pos_scaled_tanh, scaled_sigmoid, centred_linear, get_mesh_normals, load_smpl_params, get_pc, get_sin_metric, emb_init_weights


def OptLearnerStaticArchitecture(param_trainable, init_wrapper, emb_size=1000):
    """ Optimised learner network architecture """
    # An embedding layer is required to optimise the parameters
    optlearner_input = Input(shape=(1,), name="embedding_index")

    # Initialise the embedding layers
    emb_layers = init_emb_layers(optlearner_input, emb_size, param_trainable, init_wrapper)
    optlearner_params = Concatenate(name="parameter_embedding")(emb_layers)
    optlearner_params = Reshape(target_shape=(85,), name="learned_params")(optlearner_params)
    print("optlearner parameters shape: " +str(optlearner_params.shape))
    #exit(1)

    # Ground truth parameters and point cloud are inputs to the model as well
    gt_params = Input(shape=(85,), name="gt_params")
    gt_pc = Input(shape=(6890, 3), name="gt_pc")
    print("gt parameters shape: " + str(gt_params.shape))
    print("gt point cloud shape: " + str(gt_pc.shape))

    # Compute the true offset (i.e. difference) between the ground truth and learned parameters
    delta_d = Lambda(lambda x: x[0] - x[1], name="delta_d")([gt_params, optlearner_params])
    print("delta_d shape: " + str(delta_d.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned parameters and the ground truth parameters
    false_loss_delta_d = Lambda(lambda x: K.mean(K.square(x), axis=1))(delta_d)
    print("delta_d loss shape: " + str(false_loss_delta_d.shape))
    #exit(1)
    false_loss_delta_d = Reshape(target_shape=(1,), name="delta_d_mse")(false_loss_delta_d)
    print("delta_d loss shape: " + str(false_loss_delta_d.shape))

    # Load SMPL model and get necessary parameters
    smpl_params = load_params('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    _, _, input_info = get_parameters()
    input_betas = Lambda(lambda x: x[:, 72:82])(optlearner_params)
    input_pose_rodrigues = Lambda(lambda x: x[:, 0:72])(optlearner_params)
    input_trans = Lambda(lambda x: x[:, 82:85])(optlearner_params)

    # Get the point cloud corresponding to these parameters
    optlearner_pc = Points3DFromSMPLParams(input_betas, input_pose_rodrigues, input_trans, smpl_params, input_info)
    print("optlearner point cloud shape: " + str(optlearner_pc.shape))
    #exit(1)

    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_diff = Lambda(lambda x: x[0] -  x[1])([gt_pc, optlearner_pc])
    pc_euclidean_dist = Lambda(lambda x: K.sum(K.square(x),axis=-1))(pc_euclidean_diff)
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))

    # Learn the offset in parameters from the difference between the ground truth and learned point clouds
    pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    print("shape of output: " + str(pc_euclidean_diff_NOGRAD.shape))
    vertices = [1850, 1600, 2050, 1300, 5350, 5050, 5500]
    pc_euclidean_diff_NOGRAD = Lambda(lambda x: K.tf.gather(x, np.array(vertices).astype(np.int32), axis=-2))(pc_euclidean_diff_NOGRAD)
    print("shape of output: " + str(pc_euclidean_diff_NOGRAD.shape))
    #exit(1)
    pc_euclidean_diff_NOGRAD = Flatten()(pc_euclidean_diff_NOGRAD)

    optlearner_architecture = Dense(2**11, activation="relu")(pc_euclidean_diff_NOGRAD)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    print('optlearner_architecture '+str(optlearner_architecture.shape))
    #exit(1)
    delta_d_hat = Dense(85, activation="linear", name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape '+str(delta_d_hat.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    false_loss_delta_d_hat = Lambda(lambda x: K.sum(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
    false_sin_loss_delta_d_hat = get_sin_metric(delta_d_NOGRAD, delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Lambda(lambda x: x[1]*x[0], name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat, false_loss_smpl, delta_d, delta_d_hat, delta_d_hat_NOGRAD]

