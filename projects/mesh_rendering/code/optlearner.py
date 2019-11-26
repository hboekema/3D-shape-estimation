import sys
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, Flatten, Conv2D, Lambda, Concatenate, Dropout, BatchNormalization, MaxPooling2D, AveragePooling2D, Embedding, Reshape
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


# Define custom loss functions
def false_loss(y_true, y_pred):
    """ Loss function that simply return the predicted value (i.e. target is 0) """
    return y_pred

def no_loss(y_true, y_pred):
    return tf.reduce_sum(tf.multiply(y_true, 0))

def OptLearnerArchitecture(parameter_initializer=RandomUniform(minval=-0.2, maxval=0.2, seed=10)):
    """ Optimised learner network architecture """
    # An embedding layer is required to optimise the parameters
    optlearner_input = Input(shape=(1,), name="embedding_index")
    optlearner_params = Embedding(1000, 85, embeddings_initializer=parameter_initializer, name="parameter_embedding")(optlearner_input)
    optlearner_params = Reshape(target_shape=(85,), name="learned_params")(optlearner_params)
    print("optlearner parameters shape: " +str(optlearner_params.shape))

    # Ground truth parameters and point cloud are inputs to the model as well
    gt_params = Input(shape=(85,), name="gt_params")
    gt_pc = Input(shape=(6890, 3), name="gt_pc")
    print("gt parameters shape: " + str(gt_params.shape))
    print("gt point cloud shape: " + str(gt_pc.shape))

    # Compute the true offset (i.e. difference) between the ground truth and learned parameters
    delta_d = Lambda(lambda x: tf.subtract(x[0], x[1]), name="delta_d")([gt_params, optlearner_params])
    print("delta_d shape: " + str(delta_d.shape))

    # Calculate the (batched) MSE between the learned parameters and the ground truth parameters
    false_loss_delta_d = Lambda(lambda x: tf.reduce_mean(tf.square(x), axis=1))(delta_d)
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

#    # Get the (batched) MSE between the learned and ground truth point clouds
#    false_loss_pc = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=[1,2]))([gt_pc, optlearner_pc])
#    false_loss_pc = Reshape(target_shape=(1,), name="pointcloud_mse")(false_loss_pc)
#    print("point cloud loss shape: " + str(false_loss_pc.shape))
    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_dist = Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.squared_difference(x[0], x[1]), axis=2)))([gt_pc, optlearner_pc])
    false_loss_pc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))

    # Learn the offset in parameters from the difference between the ground truth and learned point clouds
    flattened_gt_pc = Flatten()(gt_pc)
    flattened_optlearner_pc = Flatten()(optlearner_pc)

    concat_pc_inputs = Concatenate()([flattened_gt_pc, flattened_optlearner_pc])
    optlearner_architecture = Dense(256, activation="relu")(concat_pc_inputs)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    optlearner_architecture = Dense(1024, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    delta_d_hat = Dense(85, activation="tanh")(optlearner_architecture)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    false_loss_delta_d_hat = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=1))([delta_d, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Lambda(lambda x: tf.multiply(x[0], x[1]), name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_loss_smpl]


def OptLearnerExtraOutputArchitecture(parameter_initializer=RandomUniform(minval=-0.2, maxval=0.2, seed=10)):
    """ Optimised learner network architecture """
    # An embedding layer is required to optimise the parameters
    #print('parameter initializer pose '+str(parameter_initializer([1000,85])[0,:15]))
    #print('parameter initializer shape '+str(parameter_initializer([1000,85])[0,72:82]))
    #print('parameter initializer T '+str(parameter_initializer([1000,85])[0,82:85]))
    #exit(1)

    optlearner_input = Input(shape=(1,), name="embedding_index")
    optlearner_params = Embedding(1000, 85, embeddings_initializer=parameter_initializer, name="parameter_embedding")(optlearner_input)
    print("optlearner parameters shape: " +str(optlearner_params.shape))
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
#    # Get the (batched) MSE between the learned and ground truth point clouds
#    false_loss_pc = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=[1,2]))([gt_pc, optlearner_pc])
#    false_loss_pc = Reshape(target_shape=(1,), name="pointcloud_mse")(false_loss_pc)
#    print("point cloud loss shape: " + str(false_loss_pc.shape))
    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_dist = Lambda(lambda x: K.sum(K.square(x[0] -  x[1]), axis=2))([gt_pc, optlearner_pc])
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))

    # Learn the offset in parameters from the difference between the ground truth and learned point clouds
    #flattened_gt_pc = Flatten()(gt_pc)
    #flattened_optlearner_pc = Flatten()(optlearner_pc)

    #concat_pc_inputs = Concatenate()([flattened_gt_pc, flattened_optlearner_pc])
    pc_euclidean_dist_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_dist) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    optlearner_architecture = Dense(256, activation="relu")(pc_euclidean_dist_NOGRAD)
    print('optlearniner_architecture '+str(optlearner_architecture.shape))
    #exit(1)
    optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    optlearner_architecture = Dense(1024, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    delta_d_hat = Dense(85, activation="linear", name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape '+str(delta_d_hat.shape))
    #exit(1)
    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Lambda(lambda x: x[1]*x[0], name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_loss_smpl, delta_d, delta_d_hat,delta_d_hat_NOGRAD]


def OptLearnerStaticArchitecture(param_trainable, init_wrapper, emb_size=1000):
    """ Optimised learner network architecture """
    # An embedding layer is required to optimise the parameters
    #print('parameter initializer pose '+str(parameter_initializer([1000,85])[0,:15]))
    #print('parameter initializer shape '+str(parameter_initializer([1000,85])[0,72:82]))
    #print('parameter initializer T '+str(parameter_initializer([1000,85])[0,82:85]))
    #exit(1)

    optlearner_input = Input(shape=(1,), name="embedding_index")
    def init_emb_layers(index, param_trainable, init_wrapper):
        """ Initialise the parameter embedding layers """
        emb_layers = []
        num_params = 85
        for i in range(num_params):
            layer_name = "param_{:02d}".format(i)
            initialiser = init_wrapper(param=i, offset=param_trainable[layer_name])

            emb_layer = Embedding(emb_size, 1, name=layer_name, trainable=param_trainable[layer_name], embeddings_initializer=initialiser)(index)
            emb_layers.append(emb_layer)

        return emb_layers

    # Initialise the embedding layers
    emb_layers = init_emb_layers(optlearner_input, param_trainable, init_wrapper)
    optlearner_params = Concatenate(name="parameter_embedding")(emb_layers)

    #optlearner_params = Embedding(1000, 85, embeddings_initializer=parameter_initializer, name="parameter_embedding")(optlearner_input)
    #print("optlearner parameters shape: " +str(optlearner_params.shape))
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
    #optlearner_pc = Lambda(lambda x: x * 0.0)[optlearner_pc]

    #exit(1)
#    # Get the (batched) MSE between the learned and ground truth point clouds
#    false_loss_pc = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=[1,2]))([gt_pc, optlearner_pc])
#    false_loss_pc = Reshape(target_shape=(1,), name="pointcloud_mse")(false_loss_pc)
#    print("point cloud loss shape: " + str(false_loss_pc.shape))
    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_diff = Lambda(lambda x: x[0] -  x[1])([gt_pc, optlearner_pc])
    pc_euclidean_dist = Lambda(lambda x: K.sum(K.square(x),axis=-1))(pc_euclidean_diff)
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))

    # Learn the offset in parameters from the difference between the ground truth and learned point clouds
    #flattened_gt_pc = Flatten()(gt_pc)
    #flattened_optlearner_pc = Flatten()(optlearner_pc)
    #concat_pc_inputs = Concatenate()([flattened_gt_pc, flattened_optlearner_pc])
    pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    print("shape of output: " + str(pc_euclidean_diff_NOGRAD.shape))
    index_list=[1850, 1600, 2050, 5350, 5050, 5500]
    item_list = []
    for id in index_list:
	item = Lambda(lambda x: x[:,id:(id+1), :])(pc_euclidean_diff_NOGRAD)
        item_list.append(item)
    pc_euclidean_diff_NOGRAD = Concatenate(axis=-2)(item_list)
    #pc_euclidean_diff_NOGRAD = Lambda(lambda x: x[:, [1850, 1600, 2050, 5350, 5050, 5500], :])(pc_euclidean_diff_NOGRAD)
    print("shape of output: " + str(pc_euclidean_diff_NOGRAD.shape))
    #exit(1)
    pc_euclidean_diff_NOGRAD = Flatten()(pc_euclidean_diff_NOGRAD)
    #pc_euclidean_diff_NOGRAD = Lambda(lambda x: x[:, 5020:5600])(pc_euclidean_diff_NOGRAD)  # restrict the points the network sees to make the learning task easier
    #pc_diff_right_arm = Lambda(lambda x: x[:, 5400:5600])(pc_euclidean_diff_NOGRAD)  # restrict the points the network sees to make the learning task easier
    #pc_diff_left_arm = Lambda(lambda x: x[:, 1850:2050])(pc_euclidean_diff_NOGRAD)  # restrict the points the network sees to make the learning task easier
    #pc_euclidean_diff_NOGRAD = Concatenate()([pc_diff_right_arm, pc_diff_left_arm])
    #pc_diff1 = Lambda(lambda x: x[:, 5350])(pc_euclidean_diff_NOGRAD)
    #pc_diff2 = Lambda(lambda x: x[:, 5050])(pc_euclidean_diff_NOGRAD)
    #pc_diff3 = Lambda(lambda x: x[:, 5500])(pc_euclidean_diff_NOGRAD)
    #pc_euclidean_diff_NOGRAD = Concatenate()([pc_diff1, pc_diff2, pc_diff3])

    #optlearner_architecture = Dense(2**12, activation="relu")(pc_euclidean_diff_NOGRAD)
    optlearner_architecture = Dense(2**11, activation="relu")(pc_euclidean_diff_NOGRAD)
    #optlearner_architecture = Dense(2**10, activation="relu")(pc_euclidean_diff_NOGRAD)
    print('optlearner_architecture '+str(optlearner_architecture.shape))
    #exit(1)
    #optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    #optlearner_architecture = Dense(1024, activation="relu")(optlearner_architecture)
    #optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    delta_d_hat = Dense(85, activation="linear", name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape '+str(delta_d_hat.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d, delta_d_hat])
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    false_loss_delta_d_hat = Lambda(lambda x: K.sum(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Lambda(lambda x: x[1]*x[0], name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_loss_smpl, delta_d, delta_d_hat,delta_d_hat_NOGRAD]


def OptLearnerDistArchitecture(parameter_initializer=RandomUniform(minval=-0.2, maxval=0.2, seed=10)):
    """ Optimised learner network architecture """
    # An embedding layer is required to optimise the parameters
    optlearner_input = Input(shape=(1,), name="embedding_index")
    optlearner_params = Embedding(1000, 85, embeddings_initializer=parameter_initializer, name="parameter_embedding")(optlearner_input)
    optlearner_params = Reshape(target_shape=(85,), name="learned_params")(optlearner_params)
    print("optlearner parameters shape: " +str(optlearner_params.shape))

    # Ground truth parameters and point cloud are inputs to the model as well
    gt_params = Input(shape=(85,), name="gt_params")
    gt_pc = Input(shape=(6890, 3), name="gt_pc")
    print("gt parameters shape: " + str(gt_params.shape))
    print("gt point cloud shape: " + str(gt_pc.shape))

    # Compute the true offset (i.e. difference) between the ground truth and learned parameters
    delta_d = Lambda(lambda x: tf.subtract(x[0], x[1]), name="delta_d")([gt_params, optlearner_params])
    print("delta_d shape: " + str(delta_d.shape))

    # Calculate the (batched) MSE between the learned parameters and the ground truth parameters
    false_loss_delta_d = Lambda(lambda x: tf.reduce_mean(tf.square(x), axis=1))(delta_d)
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

    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_dist = Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.squared_difference(x[0], x[1]), axis=2)))([gt_pc, optlearner_pc])
    false_loss_pc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))

    # Learn the offset in parameters from the difference between the ground truth and learned point clouds
    optlearner_architecture = Dense(512, activation="relu")(pc_euclidean_dist)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    optlearner_architecture = Dense(1024, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    delta_d_hat = Dense(85, activation="tanh")(optlearner_architecture)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    false_loss_delta_d_hat = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=1))([delta_d, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Lambda(lambda x: tf.multiply(x[0], x[1]), name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_loss_smpl]


def OptLearnerSingleInputArchitecture(parameter_initializer=RandomUniform(minval=-0.2, maxval=0.2, seed=10)):
    """ Optimised learner network architecture """
    # An embedding layer is required to optimise the parameters

    optlearner_input = Input(shape=(1,), name="embedding_index")
    optlearner_params = Embedding(1000, 85, embeddings_initializer=parameter_initializer, name="parameter_embedding")(optlearner_input)
    optlearner_params = Reshape(target_shape=(85,), name="learned_params")(optlearner_params)
    print("optlearner parameters shape: " +str(optlearner_params.shape))

    optlearner_input = Input(shape=(85,))

    # Ground truth parameters and point cloud are inputs to the model as well
    gt_params = Input(shape=(85,), name="gt_params")
    gt_pc = Input(shape=(6890, 3), name="gt_pc")
    print("gt parameters shape: " + str(gt_params.shape))
    print("gt point cloud shape: " + str(gt_pc.shape))

    # Compute the true offset (i.e. difference) between the ground truth and learned parameters
    delta_d = Lambda(lambda x: tf.subtract(x[0], x[1]), name="delta_d")([gt_params, optlearner_input])
    print("delta_d shape: " + str(delta_d.shape))

    # Calculate the (batched) MSE between the learned parameters and the ground truth parameters
    false_loss_delta_d = Lambda(lambda x: tf.reduce_mean(tf.square(x), axis=1))(delta_d)
    false_loss_delta_d = Reshape(target_shape=(1,), name="delta_d_mse")(false_loss_delta_d)
    print("delta_d loss shape: " + str(false_loss_delta_d.shape))

    # Load SMPL model and get necessary parameters
    smpl_params = load_params('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    _, _, input_info = get_parameters()
    input_betas = Lambda(lambda x: x[:, 72:82])(optlearner_input)
    input_pose_rodrigues = Lambda(lambda x: x[:, 0:72])(optlearner_input)
    input_trans = Lambda(lambda x: x[:, 82:85])(optlearner_input)

    # Get the point cloud corresponding to these parameters
    optlearner_pc = Points3DFromSMPLParams(input_betas, input_pose_rodrigues, input_trans, smpl_params, input_info)
    print("optlearner point cloud shape: " + str(optlearner_pc.shape))

#    # Get the (batched) MSE between the learned and ground truth point clouds
#    false_loss_pc = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=[1,2]))([gt_pc, optlearner_pc])
#    false_loss_pc = Reshape(target_shape=(1,), name="pointcloud_mse")(false_loss_pc)
#    print("point cloud loss shape: " + str(false_loss_pc.shape))
    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_dist = Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.squared_difference(x[0], x[1]), axis=2)))([gt_pc, optlearner_pc])
    false_loss_pc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))

    # Learn the offset in parameters from the difference between the ground truth and learned point clouds
    flattened_gt_pc = Flatten()(gt_pc)
    flattened_optlearner_pc = Flatten()(optlearner_pc)

    concat_pc_inputs = Concatenate()([flattened_gt_pc, flattened_optlearner_pc])
    optlearner_architecture = Dense(256, activation="relu")(concat_pc_inputs)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    optlearner_architecture = Dense(1024, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    delta_d_hat = Dense(85, activation="tanh")(optlearner_architecture)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    false_loss_delta_d_hat = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=1))([delta_d, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Lambda(lambda x: tf.multiply(x[0], x[1]), name="smpl_diff")([optlearner_input, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_input, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_loss_smpl]

def NormLearnerArchitecture(parameter_initializer=RandomUniform(minval=-0.2, maxval=0.2, seed=10)):
    """ Normal learner network architecture """
    # An embedding layer is required to optimise the parameters
    normlearner_input = Input(shape=(1,), name="embedding_index")
    normlearner_params = Embedding(1000, 85, embeddings_initializer=parameter_initializer)(normlearner_input)
    normlearner_params = Reshape(target_shape=(85,), name="learned_params")(normlearner_params)
    print("normlearner parameters shape: " +str(normlearner_params.shape))

    # Ground truth parameters and point cloud are inputs to the model as well
    gt_params = Input(shape=(85,), name="gt_params")
    gt_pc = Input(shape=(6890, 3), name="gt_pc")
    print("gt parameters shape: " + str(gt_params.shape))
    print("gt point cloud shape: " + str(gt_pc.shape))

    # Compute the true offset (i.e. difference) between the ground truth and learned parameters
    delta_d = Lambda(lambda x: tf.subtract(x[0], x[1]), name="delta_d")([gt_params, normlearner_params])
    print("delta_d shape: " + str(delta_d.shape))

    # Calculate the (batched) MSE between the learned parameters and the ground truth parameters
    false_loss_delta_d = Lambda(lambda x: tf.reduce_mean(tf.square(x), axis=1))(delta_d)
    false_loss_delta_d = Reshape(target_shape=(1,), name="delta_d_mse")(false_loss_delta_d)
    print("delta_d loss shape: " + str(false_loss_delta_d.shape))

    # Load SMPL model and get necessary parameters
    smpl_params = load_params('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    _, _, input_info = get_parameters()
    input_betas = Lambda(lambda x: x[:, 72:82])(normlearner_params)
    input_pose_rodrigues = Lambda(lambda x: x[:, 0:72])(normlearner_params)
    input_trans = Lambda(lambda x: x[:, 82:85])(normlearner_params)

    # Get the point cloud corresponding to these parameters
    normlearner_pc = Points3DFromSMPLParams(input_betas, input_pose_rodrigues, input_trans, smpl_params, input_info)
    print("normlearner point cloud shape: " + str(normlearner_pc.shape))

    # Get the (batched) MSE between the learned and ground truth point clouds
    false_loss_pc = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=[1,2]))([gt_pc, normlearner_pc])
    false_loss_pc = Reshape(target_shape=(1,), name="pointcloud_mse")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))

#    # Learn the offset in parameters from the difference between the ground truth and learned point clouds
#    flattened_gt_pc = Flatten()(gt_pc)
#    flattened_normlearner_pc = Flatten()(normlearner_pc)

#    concat_pc_inputs = Concatenate()([flattened_gt_pc, flattened_normlearner_pc])
#    normlearner_architecture = Dense(256, activation="relu")(concat_pc_inputs)
#    normlearner_architecture = Dense(1024, activation="relu")(normlearner_architecture)
#    delta_d_hat = Dense(85, activation="tanh")(normlearner_architecture)

#    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
#   false_loss_delta_d_hat = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=1))([delta_d, delta_d_hat])
#    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
#    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))

    return [normlearner_input, gt_params, gt_pc], [normlearner_params, false_loss_delta_d, normlearner_pc, false_loss_pc]
