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


# Define custom loss functions
def false_loss(y_true, y_pred):
    """ Loss function that simply return the predicted value (i.e. target is 0) """
    return y_pred

def no_loss(y_true, y_pred):
    return tf.reduce_sum(tf.multiply(y_true, 0))

def cat_xent(true_indices, y_pred):
    pred_probs = K.tf.gather_nd(y_pred, true_indices, batch_dims=1)
    print("pred_probs shape: " + str(pred_probs.shape))
    return -K.tf.math.log(pred_probs)

def mape(y_true, y_pred):
    """ Calculate the mean absolute percentage error """
    epsilon = 1e-3
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), epsilon, None))
    return 100. * K.mean(diff, axis=-1)

# Custom activation functions
def scaled_tanh(x):
    """ Tanh scaled by pi """
    return K.tf.constant(np.pi) * K.tanh(x)

def pos_scaled_tanh(x):
    """ Sigmoid scaled by 2 pi """
    return K.tf.constant(np.pi) * (K.tanh(x) + 1)

def scaled_sigmoid(x):
    """ Sigmoid scaled by 2 pi """
    return K.tf.constant(2*np.pi) * K.sigmoid(x)

def centred_linear(x):
    """ Linear shifted by pi """
    return K.tf.constant(np.pi) + x


# Custom functions
def get_mesh_normals(pc, faces, layer_name="mesh_normals"):
    """ Gather sets of points and compute their cross product to get mesh normals """
    p0 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,0]).astype(np.int32), axis=-2))(pc)
    p1 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,1]).astype(np.int32), axis=-2))(pc)
    p2 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,2]).astype(np.int32), axis=-2))(pc)
    print("p0 shape: " + str(p0.shape))
    print("p1 shape: " + str(p1.shape))
    print("p2 shape: " + str(p2.shape))
    vec1 = Lambda(lambda x: x[1] - x[0])([p0, p1])
    vec2 = Lambda(lambda x: x[1] - x[0])([p0, p2])
    print("vec1 shape: " + str(vec1.shape))
    print("vec2 shape: " + str(vec2.shape))
    normals = Lambda(lambda x: K.l2_normalize(K.tf.cross(x[0], x[1]), axis=-1), name=layer_name)([vec1, vec2])

    return normals

def load_smpl_params():
    # Load SMPL model and get necessary parameters
    smpl_params = load_params('./keras_rotationnet_v2_demo_for_hidde/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    _, _, input_info = get_parameters()
    faces = smpl_params['f']    # canonical mesh faces
    print("faces shape: " + str(faces.shape))
    #exit(1)
    return smpl_params, input_info, faces

def get_pc(optlearner_params, smpl_params, input_info, faces):
    """ Use the SMPL model to render parameters in to point clouds"""
    input_betas = Lambda(lambda x: x[:, 72:82])(optlearner_params)
    input_pose_rodrigues = Lambda(lambda x: x[:, 0:72])(optlearner_params)
    input_trans = Lambda(lambda x: x[:, 82:85])(optlearner_params)

    # Get the point cloud corresponding to these parameters
    optlearner_pc = Points3DFromSMPLParams(input_betas, input_pose_rodrigues, input_trans, smpl_params, input_info)
    #optlearner_pc = Lambda(lambda x: Points3DFromSMPLParams(x[0], x[1], x[2], smpl_params, input_info))([input_betas, input_pose_rodrigues, input_trans])
    print("optlearner point cloud shape: " + str(optlearner_pc.shape))
    return optlearner_pc

def get_sin_metric(delta_d, delta_d_hat):
    """ Calculate the sin metric for evaluating different architectures """
    #false_sin_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square((x[0] - x[1]) * K.tf.sin(0.5*(x[0] - x[1]))), axis=1))([delta_d, delta_d_hat])
    false_sin_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(K.tf.sin(0.5*(x[0] - x[1]))), axis=1))([delta_d, delta_d_hat])
    false_sin_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_sin_mse")(false_sin_loss_delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))
    return false_sin_loss_delta_d_hat

def init_emb_layers(index, emb_size, param_trainable, init_wrapper):
        """ Initialise the parameter embedding layers """
        emb_layers = []
        num_params = 85
        for i in range(num_params):
            layer_name = "param_{:02d}".format(i)
            initialiser = init_wrapper(param=i, offset=param_trainable[layer_name])

            emb_layer = Embedding(emb_size, 1, name=layer_name, trainable=param_trainable[layer_name], embeddings_initializer=initialiser)(index)
            emb_layers.append(emb_layer)

        return emb_layers

def emb_init_weights(emb_params, period=None):
    """ Embedding weights initialiser """
    def emb_init_wrapper(param, offset=False):
        def emb_init(shape):
            """ Initializer for the embedding layer """
            curr_emb_param = K.tf.gather(emb_params, param, axis=-1)
            curr_emb_param = K.tf.cast(curr_emb_param, dtype="float32")

            if offset:
                k = K.constant(np.pi)
                offset_value = K.random_uniform(shape=[shape[0]], minval=-k, maxval=k, dtype="float32")
                #print(offset_value)
                #exit(1)
                if period is not None and shape[0] % period == 0:
                    block_size = shape[0] // period
                    #factors = Concatenate()([K.random_normal(shape=[block_size], mean=np.sqrt((i+1)/period), stddev=0.01) for i in range(period)])
                    factors = Concatenate()([K.random_normal(shape=[block_size], mean=np.sqrt(float(i+1)/period), stddev=0.01) for i in range(period)])
                    offset_value = Multiply()([offset_value, factors])

                curr_emb_param = Add()([curr_emb_param, offset_value])

            init = Reshape(target_shape=[shape[1]])(curr_emb_param)
            #print("init shape: " + str(init.shape))
            #exit(1)
            return init
        return emb_init
    return emb_init_wrapper

def custom_mod(x, k=K.constant(np.pi), name="custom_mod"):
    """ Custom mod function """
    y = Lambda(lambda x: K.tf.math.floormod(x - k, 2*k) - k, name=name)(x)
    return y


# Network architectures
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
    vertices = [1850, 1600, 2050, 5350, 5050, 5500]
    pc_euclidean_diff_NOGRAD = Lambda(lambda x: K.tf.gather(x, np.array(vertices).astype(np.int32), axis=-2))(pc_euclidean_diff_NOGRAD)
    print("shape of output: " + str(pc_euclidean_diff_NOGRAD.shape))
    #exit(1)
    pc_euclidean_diff_NOGRAD = Flatten()(pc_euclidean_diff_NOGRAD)

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


def OptLearnerStaticModArchitecture(param_trainable, init_wrapper, smpl_params, input_info, faces, emb_size=1000):
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
    pi = K.constant(np.pi)
    delta_d = Lambda(lambda x: x[0] - x[1])([gt_params, optlearner_params])
    delta_d = Lambda(lambda x: K.tf.math.floormod(x - pi, 2*pi) - pi, name="delta_d")(delta_d)  # custom modulo 2pi of delta_d
    print("delta_d shape: " + str(delta_d.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned parameters and the ground truth parameters
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
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))
    #exit(1)

    # Gather sets of points and compute their difference vectors
    vertices=[1850, 1600, 2050, 1300, 5350, 5050, 5500]
    #vertices=[1850, 1600, 2050, 5350, 5050, 5500]
    diff_vec_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    print("diff_vec_NOGRAD shape: " + str(diff_vec_NOGRAD.shape))
    diff_vec_NOGRAD = Lambda(lambda x: K.tf.gather(x, np.array(vertices).astype(np.int32), axis=-2))(diff_vec_NOGRAD)
    diff_vec_NOGRAD = Flatten()(diff_vec_NOGRAD)

    optlearner_architecture = Dense(2**11, activation="relu")(diff_vec_NOGRAD)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    optlearner_architecture = Dropout(0.2)(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #delta_d_hat = Dense(85, activation=pos_scaled_tanh, name="delta_d_hat")(optlearner_architecture)
    #delta_d_hat = Dense(85, activation="linear", name="delta_d_hat")(optlearner_architecture)
    delta_d_hat = Dense(85, activation=centred_linear, name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape: '+str(delta_d_hat.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
    false_sin_loss_delta_d_hat = get_sin_metric(delta_d_NOGRAD, delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Multiply(name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat,  false_loss_smpl, delta_d, delta_d_hat, delta_d_hat_NOGRAD]


def OptLearnerStaticModBinnedArchitecture(param_trainable, init_wrapper, emb_size=1000):
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
    pi = K.constant(np.pi)
    delta_d = Lambda(lambda x: x[0] - x[1], name="delta_d")([gt_params, optlearner_params])
    delta_d = Lambda(lambda x: K.tf.math.floormod(x - pi, 2*pi) - pi)(delta_d)  # symmetric modulo 2pi of delta_d
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
    vertices = [1850, 1600, 2050, 5350, 5050, 5500]
    pc_euclidean_diff_NOGRAD = Lambda(lambda x: K.tf.gather(x, np.array(vertices).astype(np.int32), axis=-2))(pc_euclidean_diff_NOGRAD)
    print("shape of output: " + str(pc_euclidean_diff_NOGRAD.shape))
    #exit(1)
    pc_euclidean_diff_NOGRAD = Flatten()(pc_euclidean_diff_NOGRAD)

    # Bin the parameter differences
    nparams = 85
    nbins = 100
    num_params = K.constant(nparams, dtype="int32")
    num_bins = K.constant(nbins, dtype="int32")
    value_range = [-pi, pi]
    delta_d_indices = Lambda(lambda x: K.tf.histogram_fixed_width_bins(x, value_range, num_bins))(delta_d)

    batch_dim = Lambda(lambda x: K.tf.shape(x)[0:1])(delta_d_indices)
    param_index = K.constant([i for i in range(nparams)], dtype="int32")
    print("param_index shape: " + str(param_index.shape))
    param_indices_ = Lambda(lambda x: K.tf.tile(x[0], x[1]))([param_index, batch_dim])
    param_dim = Lambda(lambda x: K.tf.shape(x)[0:1])(param_indices_)
    new_shape = Lambda(lambda x: K.tf.concat([x[0], x[1]], axis=0))([batch_dim, param_dim])
    #param_indices = Lambda(lambda x: K.tf.reshape(x[0], K.tf.cast(x[1], dtype=K.tf.dtypes.int32)))([param_indices, new_shape])
    param_indices = Lambda(lambda x: K.tf.reshape(x[0], K.tf.cast(x[1], dtype=K.tf.dtypes.int32)))([param_indices_, new_shape])
    print("param_indices shape: " + str(param_indices.shape))
    delta_d_indices = Lambda(lambda x: K.tf.stack(x, axis=-1))([param_indices, delta_d_indices])
    print("delta_d_indices shape: " + str(delta_d_indices.shape))
    delta_d_indices_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d_indices)
    print("delta_d_indices_NOGRAD shape: " + str(delta_d_indices_NOGRAD.shape))
    #exit(1)

    #optlearner_architecture = Dense(2**12, activation="relu")(pc_euclidean_diff_NOGRAD)
    optlearner_architecture = Dense(2**12, activation="relu")(pc_euclidean_diff_NOGRAD)
    #optlearner_architecture = Dense(2**10, activation="relu")(pc_euclidean_diff_NOGRAD)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    #optlearner_architecture = Dense(2**10, activation="relu")(optlearner_architecture)
    #optlearner_architecture = BatchNormalization()(optlearner_architecture)
    #optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    print('optlearner_architecture '+str(optlearner_architecture.shape))
    delta_d_hat_layers = []
    for param in range(nparams):
            layer_name = "param_{:02d}_pred".format(param)
            param_layer = Dense(nbins, activation="softmax", name=layer_name)(optlearner_architecture)
            delta_d_hat_layers.append(param_layer)
            #print("param_layer shape: " + str(param_layer.shape))
            #exit(1)

    delta_d_hat_binned = Lambda(lambda x: K.tf.stack(x, axis=1), name="parameter_preds")(delta_d_hat_layers)
    print('delta_d_hat_binned shape '+str(delta_d_hat_binned.shape))

    # Get the categorical cross-entropy loss between the learned and ground truth parameter offsets
    #xent_loss_delta_d_hat = cat_xent(delta_d_indices, delta_d_hat_binned)
    #print("xent_loss_delta_d_hat shape: " + str(xent_loss_delta_d_hat.shape))
    #exit(1)

    # Retrieve the delta_d_hat values
    bins = Lambda(lambda x: K.tf.linspace(x[0], x[1], K.tf.cast(x[2], dtype="int32")))([value_range[0], value_range[1], num_bins])
    print("bins shape: " + str(bins.shape))
    bins_expanded = Lambda(lambda x: K.tf.expand_dims(x, axis=0))(bins)
    bins_tiled = Lambda(lambda x: K.tf.tile(x[0], [x[1][0], x[2]]))([bins_expanded, batch_dim, num_params])
    bins_dim = Lambda(lambda x: K.tf.shape(x)[0:1])(bins)
    bin_shape = Lambda(lambda x: K.tf.concat([x[0], K.tf.reshape(x[1], [1,]), x[2]], axis=0))([batch_dim, num_params, bins_dim])
    bins = Lambda(lambda x: K.tf.reshape(x[0], K.tf.cast(x[1], dtype="int32")))([bins_tiled, bin_shape])
    print("bins (tiled) shape: " + str(bins.shape))
    delta_d_hat_indices = Lambda(lambda x: K.tf.math.argmax(x, axis=-1, output_type="int32"))(delta_d_hat_binned)
    delta_d_hat_indices = Lambda(lambda x: K.tf.stack(x, axis=-1), dtype="int32")([param_indices, delta_d_hat_indices])
    print("delta_d_hat_indices shape: " + str(delta_d_hat_indices.shape))
    delta_d_hat = Lambda(lambda x: K.tf.gather_nd(x[0], K.tf.cast(x[1], K.tf.dtypes.int32), batch_dims=1), name="delta_d_hat")([bins, delta_d_hat_indices])
    print("delta_d_hat shape: " + str(delta_d_hat.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d, delta_d_hat])
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(K.tf.math.floormod(x[0], 2*K.tf.constant(np.pi)) - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.abs(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
    false_sin_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(K.tf.sin(0.5*(x[0] - x[1]))), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_sin_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_sin_mse")(false_sin_loss_delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    #false_loss_smpl = Lambda(lambda x: x[1]*x[0], name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    false_loss_smpl = Multiply(name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

#    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat, false_loss_smpl, delta_d, delta_d_hat, delta_d_hat_NOGRAD]
    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, param_indices_]


def OptLearnerStaticSinArchitecture(param_trainable, init_wrapper, emb_size=1000):
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
    faces = smpl_params['f']    # canonical mesh faces
    print("faces shape: " + str(faces.shape))
    #exit(1)

    input_betas = Lambda(lambda x: x[:, 72:82])(optlearner_params)
    input_pose_rodrigues = Lambda(lambda x: x[:, 0:72])(optlearner_params)
    input_trans = Lambda(lambda x: x[:, 82:85])(optlearner_params)

    # Get the point cloud corresponding to these parameters
    optlearner_pc = Points3DFromSMPLParams(input_betas, input_pose_rodrigues, input_trans, smpl_params, input_info)
    print("optlearner point cloud shape: " + str(optlearner_pc.shape))
    #optlearner_pc = Lambda(lambda x: x * 0.0)[optlearner_pc]

    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_diff = Lambda(lambda x: x[0] -  x[1])([gt_pc, optlearner_pc])
    pc_euclidean_dist = Lambda(lambda x: K.sum(K.square(x),axis=-1))(pc_euclidean_diff)
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))
    #exit(1)

    # Learn the offset in parameters from the difference between the ground truth and learned point clouds
    #flattened_gt_pc = Flatten()(gt_pc)
    #flattened_optlearner_pc = Flatten()(optlearner_pc)
    #concat_pc_inputs = Concatenate()([flattened_gt_pc, flattened_optlearner_pc])
    pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    print("shape of output: " + str(pc_euclidean_diff_NOGRAD.shape))
    vertices = [1850, 1600, 2050, 5350, 5050, 5500]
    pc_euclidean_diff_NOGRAD = Lambda(lambda x: K.tf.gather(x, np.array(vertices).astype(np.int32), axis=-2))(pc_euclidean_diff_NOGRAD)
    print("shape of output: " + str(pc_euclidean_diff_NOGRAD.shape))
    #exit(1)
    pc_euclidean_diff_NOGRAD = Flatten()(pc_euclidean_diff_NOGRAD)

    optlearner_architecture = Dense(2**11, activation="relu")(pc_euclidean_diff_NOGRAD)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    optlearner_architecture = Dense(2**10, activation="relu")(optlearner_architecture)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    #optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    #optlearner_architecture = Dense(2**9, activation="relu")(optlearner_architecture)
    #optlearner_architecture = BatchNormalization()(optlearner_architecture)
    #optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #exit(1)
    #optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    #optlearner_architecture = Dense(1024, activation="relu")(optlearner_architecture)
    #optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    delta_d_hat = Dense(85, activation=scaled_tanh, name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape: '+str(delta_d_hat.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d, delta_d_hat])
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    #false_loss_delta_d_hat = Lambda(lambda x: K.sum(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
    false_sin_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square((x[0] - x[1]) * K.tf.sin(0.5*(x[0] - x[1]))), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_sin_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_sin_mse")(false_sin_loss_delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Lambda(lambda x: x[1]*x[0], name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat, false_loss_smpl, delta_d, delta_d_hat,delta_d_hat_NOGRAD]


def OptLearnerMeshNormalStaticArchitecture(param_trainable, init_wrapper, emb_size=1000):
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
    #pi = K.constant(np.pi)
    #delta_d = Lambda(lambda x: x[0] - x[1])([gt_params, optlearner_params])
    #delta_d = Lambda(lambda x: K.tf.math.floormod(x - pi, 2*pi) - pi, name="delta_d")(delta_d)  # custom modulo 2pi of delta_d
    #print("delta_d shape: " + str(delta_d.shape))
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
    faces = smpl_params['f']    # canonical mesh faces
    print("faces shape: " + str(faces.shape))
    #exit(1)

    input_betas = Lambda(lambda x: x[:, 72:82])(optlearner_params)
    input_pose_rodrigues = Lambda(lambda x: x[:, 0:72])(optlearner_params)
    input_trans = Lambda(lambda x: x[:, 82:85])(optlearner_params)

    # Get the point cloud corresponding to these parameters
    optlearner_pc = Points3DFromSMPLParams(input_betas, input_pose_rodrigues, input_trans, smpl_params, input_info)
    print("optlearner point cloud shape: " + str(optlearner_pc.shape))
    #optlearner_pc = Lambda(lambda x: x * 0.0)[optlearner_pc]

    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_diff = Lambda(lambda x: x[0] -  x[1])([gt_pc, optlearner_pc])
    pc_euclidean_dist = Lambda(lambda x: K.sum(K.square(x),axis=-1))(pc_euclidean_diff)
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))
    #exit(1)

#    # Get the (batched) MSE between the learned and ground truth point clouds
#    false_loss_pc = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=[1,2]))([gt_pc, optlearner_pc])
#    false_loss_pc = Reshape(target_shape=(1,), name="pointcloud_mse")(false_loss_pc)
#    print("point cloud loss shape: " + str(false_loss_pc.shape))

    # Gather sets of points and compute their cross product to get mesh normals
    vertex_list=[1850, 1600, 2050, 5350, 5050, 5500]
    face_array = np.array([face for face in faces for vertex in vertex_list if vertex in face])
    #gt_p0 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,0]).astype(np.int32), axis=-2))(gt_pc)
    #gt_p1 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,1]).astype(np.int32), axis=-2))(gt_pc)
    #gt_p2 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,2]).astype(np.int32), axis=-2))(gt_pc)
    #print("gt_p0 shape: " + str(gt_p0.shape))
    #print("gt_p1 shape: " + str(gt_p1.shape))
    #print("gt_p2 shape: " + str(gt_p2.shape))
    #gt_vec1 = Lambda(lambda x: x[1] - x[0])([gt_p0, gt_p1])
    #gt_vec2 = Lambda(lambda x: x[1] - x[0])([gt_p0, gt_p2])
    #print("gt_vec1 shape: " + str(gt_vec1.shape))
    #print("gt_vec2 shape: " + str(gt_vec2.shape))
    #gt_normals = Lambda(lambda x: K.l2_normalize(K.tf.cross(x[0], x[1]), axis=-1), name="gt_cross_product")([gt_vec1, gt_vec2])
    #gt_normals = get_mesh_normals(gt_pc, faces, layer_name="gt_cross_product")
    gt_normals = get_mesh_normals(gt_pc, face_array, layer_name="gt_cross_product")
    print("gt_normals shape: " + str(gt_normals.shape))

    #opt_p0 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,0]).astype(np.int32), axis=-2))(optlearner_pc)
    #opt_p1 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,1]).astype(np.int32), axis=-2))(optlearner_pc)
    #opt_p2 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,2]).astype(np.int32), axis=-2))(optlearner_pc)
    #print("opt_p0 shape: " + str(opt_p0.shape))
    #print("opt_p1 shape: " + str(opt_p1.shape))
    #print("opt_p2 shape: " + str(opt_p2.shape))
    #opt_vec1 = Lambda(lambda x: x[1] - x[0])([opt_p0, opt_p1])
    #opt_vec2 = Lambda(lambda x: x[1] - x[0])([opt_p0, opt_p2])
    #print("opt_vec1 shape: " + str(opt_vec1.shape))
    #print("opt_vec2 shape: " + str(opt_vec2.shape))
    #opt_normals = Lambda(lambda x: K.l2_normalize(K.tf.cross(x[0], x[1]), axis=-1), name="opt_cross_product")([opt_vec1, opt_vec2])
    #opt_normals = get_mesh_normals(optlearner_pc, faces, layer_name="opt_cross_product")
    opt_normals = get_mesh_normals(optlearner_pc, face_array, layer_name="opt_cross_product")
    print("opt_normals shape: " + str(opt_normals.shape))
    #exit(1)

    # Learn the offset in parameters from the difference between the ground truth and learned mesh normals
    diff_normals = Lambda(lambda x: K.tf.cross(x[0], x[1]), name="diff_cross_product")([gt_normals, opt_normals])
    #pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    diff_normals_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_normals) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    print("diff_normals_NOGRAD shape: " + str(diff_normals_NOGRAD.shape))

    # Get the mesh normal angle magnitudes (for evaluation)
    gt_angles = Lambda(lambda x: K.tf.norm(x, axis=-1), name="gt_angles")(gt_normals)
    print("gt_angles shape: " + str(gt_angles.shape))
    opt_angles = Lambda(lambda x: K.tf.norm(x, axis=-1), name="opt_angles")(opt_normals)
    print("opt_angles shape: " + str(opt_angles.shape))
    diff_angles = Lambda(lambda x: K.tf.norm(x, axis=-1), name="diff_angles")(diff_normals_NOGRAD)
    print("diff_angles shape: " + str(diff_angles.shape))
    #exit(1)

    # Keep every 5xth normal entry
    #indices = np.array([i for i in enumerate()])
    #diff_normals_NOGRAD = Lambda(lambda x: x[:, ::10], name="reduce_num_normals")(diff_normals_NOGRAD)
    diff_normals_NOGRAD = Flatten()(diff_normals_NOGRAD)

    optlearner_architecture = Dense(2**11, activation="relu")(diff_normals_NOGRAD)
    #optlearner_architecture = Dense(2**12, activation="relu")(diff_normals_NOGRAD)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    #optlearner_architecture = Dense(2**10, activation="relu")(optlearner_architecture)
    #optlearner_architecture = BatchNormalization()(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #exit(1)
    #optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    #optlearner_architecture = Dense(1024, activation="relu")(optlearner_architecture)
    #optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    delta_d_hat = Dense(85, activation="linear", name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape: '+str(delta_d_hat.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d, delta_d_hat])
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name="delta_d_NOGRAD")(delta_d)
    #false_loss_delta_d_hat = Lambda(lambda x: K.sum(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Lambda(lambda x: x[1]*x[0], name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    #return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_loss_smpl, delta_d, delta_d_hat,delta_d_hat_NOGRAD]
    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_loss_smpl, delta_d, delta_d_hat,delta_d_hat_NOGRAD, gt_angles, opt_angles, diff_angles]


def OptLearnerMeshNormalStaticModArchitecture(param_trainable, init_wrapper, smpl_params, input_info, faces, emb_size=1000):
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
    pi = K.constant(np.pi)
    delta_d = Lambda(lambda x: x[0] - x[1], name="delta_d")([gt_params, optlearner_params])
    #delta_d = Lambda(lambda x: x[0] - x[1], name="delta_d_no_mod")([gt_params, optlearner_params])
    #delta_d = Lambda(lambda x: K.tf.math.floormod(x - pi, 2*pi) - pi, name="delta_d")(delta_d)  # custom modulo 2pi of delta_d
    #delta_d = custom_mod(delta_d, pi, name="delta_d")  # custom modulo 2pi of delta_d
    print("delta_d shape: " + str(delta_d.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned parameters and the ground truth parameters
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
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))
    #exit(1)

    # Gather sets of points and compute their cross product to get mesh normals
    vertex_list=[1850, 1600, 2050, 1300, 5350, 5050, 5500]
    #vertex_list=[1850, 1600, 2050, 5350, 5050, 5500]
    pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    vertex_diff_NOGRAD = Lambda(lambda x: K.tf.gather(x, np.array(vertex_list).astype(np.int32), axis=-2))(pc_euclidean_diff_NOGRAD)
    print("vertex_diff_NOGRAD shape: " + str(vertex_diff_NOGRAD.shape))
    vertex_diff_NOGRAD = Flatten()(vertex_diff_NOGRAD)
    #exit(1)
    face_array = np.array([[face for face in faces if vertex in face][0] for vertex in vertex_list])      # only take a single face for each vertex
    print("face_array shape: " + str(face_array.shape))
    gt_normals = get_mesh_normals(gt_pc, face_array, layer_name="gt_cross_product")
    print("gt_normals shape: " + str(gt_normals.shape))
    opt_normals = get_mesh_normals(optlearner_pc, face_array, layer_name="opt_cross_product")
    print("opt_normals shape: " + str(opt_normals.shape))
    #exit(1)

    # Learn the offset in parameters from the difference between the ground truth and learned mesh normals
    #diff_normals = Lambda(lambda x: K.tf.cross(x[0], x[1]), name="diff_cross_product")([gt_normals, opt_normals])
    diff_angles = Lambda(lambda x: K.tf.subtract(x[0], x[1]), name="diff_angle")([gt_normals, opt_normals])
    diff_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_angles)
    diff_angles_norm_NOGRAD = Lambda(lambda x: K.tf.norm(x, axis=-1), name="diff_angle_norm")(diff_angles_NOGRAD)
    dist_angles = Lambda(lambda x: K.mean(K.square(x), axis=-1), name="diff_angle_mse")(diff_angles)
    dist_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(dist_angles)
    print("diff_angles shape: " + str(diff_angles.shape))
    print("dist_angles shape: " + str(dist_angles.shape))
    #pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    #diff_normals_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_normals) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    #print("diff_normals_NOGRAD shape: " + str(diff_normals_NOGRAD.shape))
    #diff_normals_NOGRAD = Flatten()(diff_normals_NOGRAD)
    diff_angles_NOGRAD = Flatten()(diff_angles_NOGRAD)

    #optlearner_architecture = Dense(2**11, activation="relu")(vertex_diff_NOGRAD)
    #optlearner_architecture = Dense(2**11, activation="relu")(diff_angles_norm_NOGRAD)
    optlearner_architecture = Dense(2**11, activation="relu")(diff_angles_NOGRAD)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    #optlearner_architecture = Dropout(0.2)(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #delta_d_hat = Dense(85, activation=pos_scaled_tanh, name="delta_d_hat")(optlearner_architecture)
    delta_d_hat = Dense(85, activation="linear", name="delta_d_hat")(optlearner_architecture)
    #delta_d_hat = Dense(85, activation=centred_linear, name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape: '+str(delta_d_hat.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    #false_loss_delta_d_hat = Lambda(lambda x: K.sum(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    #false_loss_delta_d_hat = Lambda(lambda x: mape(x[0], x[1]))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
    false_sin_loss_delta_d_hat = get_sin_metric(delta_d_NOGRAD, delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Multiply(name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    #return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat,  false_loss_smpl, delta_d, delta_d_hat, delta_d_hat_NOGRAD]
    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat,  false_loss_smpl, delta_d, delta_d_hat, dist_angles]


def OptLearnerCombinedStaticModArchitecture(param_trainable, init_wrapper, smpl_params, input_info, faces, emb_size=1000):
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
    pi = K.constant(np.pi)
    #delta_d = Lambda(lambda x: x[0] - x[1], name="delta_d")([gt_params, optlearner_params])
    delta_d = Lambda(lambda x: x[0] - x[1], name="delta_d_no_mod")([gt_params, optlearner_params])
    #delta_d = Lambda(lambda x: K.tf.math.floormod(x - pi, 2*pi) - pi, name="delta_d")(delta_d)  # custom modulo 2pi of delta_d
    delta_d = custom_mod(delta_d, pi, name="delta_d")  # custom modulo 2pi of delta_d
    print("delta_d shape: " + str(delta_d.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned parameters and the ground truth parameters
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
    print('pc euclidean dist shape: '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))
    #exit(1)

    # Gather sets of points and compute their cross product to get mesh normals
    vertex_list=[1850, 1600, 2050, 1300, 5350, 5050, 5500]
    #vertex_list=[1850, 1600, 2050, 5350, 5050, 5500]
    pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    vertex_diff_NOGRAD = Lambda(lambda x: K.tf.gather(x, np.array(vertex_list).astype(np.int32), axis=-2))(pc_euclidean_diff_NOGRAD)
    print("vertex_diff_NOGRAD shape: " + str(vertex_diff_NOGRAD.shape))
    #exit(1)
    vertex_diff_NOGRAD = Flatten()(vertex_diff_NOGRAD)
    face_array = np.array([[face for face in faces if vertex in face][0] for vertex in vertex_list])      # only take a single face for each vertex
    print("face_array shape: " + str(face_array.shape))
    gt_normals = get_mesh_normals(gt_pc, face_array, layer_name="gt_cross_product")
    print("gt_normals shape: " + str(gt_normals.shape))
    opt_normals = get_mesh_normals(optlearner_pc, face_array, layer_name="opt_cross_product")
    print("opt_normals shape: " + str(opt_normals.shape))
    #exit(1)

    # Learn the offset in parameters from the difference between the ground truth and learned mesh normals
    #diff_normals = Lambda(lambda x: K.tf.cross(x[0], x[1]), name="diff_cross_product")([gt_normals, opt_normals])
    diff_angles = Lambda(lambda x: K.tf.subtract(x[0], x[1]))([gt_normals, opt_normals])
    diff_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_angles)
    dist_angles = Lambda(lambda x: K.mean(K.square(x), axis=-1), name="diff_angle_mse")(diff_angles)
    dist_angles_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(dist_angles)
    print("diff_angles shape: " + str(diff_angles.shape))
    #print("dist_angles shape: " + str(dist_angles.shape))
    #diff_normals_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_normals) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    #print("diff_normals_NOGRAD shape: " + str(diff_normals_NOGRAD.shape))
    #diff_normals_NOGRAD = Flatten()(diff_normals_NOGRAD)
    diff_angles_NOGRAD = Flatten()(diff_angles)

    # NOGRADs
    #gt_normals_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(gt_normals)
    #gt_normals_NOGRAD = Flatten()(gt_normals_NOGRAD)
    #opt_normals_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(opt_normals)
    #opt_normals_NOGRAD = Flatten()(opt_normals_NOGRAD)

    # Combine the normals mse with the point cloud differences
    pc_differences = Concatenate()([diff_angles_NOGRAD, vertex_diff_NOGRAD])
    #pc_differences = Concatenate()([dist_angles_NOGRAD, vertex_diff_NOGRAD])
    #pc_differences = Concatenate()([vertex_diff_NOGRAD, gt_normals_NOGRAD, opt_normals_NOGRAD])

    optlearner_architecture = Dense(2**11, activation="relu")(pc_differences)
    #optlearner_architecture = Dense(2**11, activation="relu")(optlearner_architecture)
    #optlearner_architecture = Dense(2**12, activation="relu")(pc_differences)
    #optlearner_architecture = Dense(2**12, activation="relu")(vertex_diff_NOGRAD)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #delta_d_hat = Dense(85, activation=pos_scaled_tanh, name="delta_d_hat")(optlearner_architecture)
    #delta_d_hat = Dense(85, activation=scaled_tanh, name="delta_d_hat")(optlearner_architecture)
    delta_d_hat = Dense(85, activation="linear", name="delta_d_hat")(optlearner_architecture)
    #delta_d_hat = Dense(85, activation=centred_linear, name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape: '+str(delta_d_hat.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    #delta_d_hat_diff = Lambda(lambda x: x[0] - x[1])([delta_d_NOGRAD, delta_d_hat])
    #delta_d_hat_mod = custom_mod(delta_d_hat_diff, pi, name="diff_mod")
    #print("delta_d_hat_mod shape: " + str(delta_d_hat_mod.shape))
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x), axis=1))(delta_d_hat_mod)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
    #delta_d_hat_se = Lambda(lambda x: K.square(x[0] - x[1]))([delta_d_NOGRAD, delta_d_hat])
    #sign_loss = Lambda(lambda x: 1.5 - 0.5 * K.sign(x[0]) * K.sign(x[1]))([delta_d_NOGRAD, delta_d_hat])
    #false_loss_delta_d_hat = Multiply()([delta_d_hat_se, sign_loss])
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(x, axis=1))(false_loss_delta_d_hat)
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.abs(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
    false_sin_loss_delta_d_hat = get_sin_metric(delta_d_NOGRAD, delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Multiply(name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat,  false_loss_smpl, delta_d, delta_d_hat, dist_angles]


def OptLearnerMeshNormalStaticSinArchitecture(param_trainable, init_wrapper, emb_size=1000):
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
    faces = smpl_params['f']    # canonical mesh faces
    print("faces shape: " + str(faces.shape))
    #exit(1)

    input_betas = Lambda(lambda x: x[:, 72:82])(optlearner_params)
    input_pose_rodrigues = Lambda(lambda x: x[:, 0:72])(optlearner_params)
    input_trans = Lambda(lambda x: x[:, 82:85])(optlearner_params)

    # Get the point cloud corresponding to these parameters
    optlearner_pc = Points3DFromSMPLParams(input_betas, input_pose_rodrigues, input_trans, smpl_params, input_info)
    print("optlearner point cloud shape: " + str(optlearner_pc.shape))
    #optlearner_pc = Lambda(lambda x: x * 0.0)[optlearner_pc]

    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_diff = Lambda(lambda x: x[0] -  x[1])([gt_pc, optlearner_pc])
    pc_euclidean_dist = Lambda(lambda x: K.sum(K.square(x),axis=-1))(pc_euclidean_diff)
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))
    #exit(1)

#    # Get the (batched) MSE between the learned and ground truth point clouds
#    false_loss_pc = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=[1,2]))([gt_pc, optlearner_pc])
#    false_loss_pc = Reshape(target_shape=(1,), name="pointcloud_mse")(false_loss_pc)
#    print("point cloud loss shape: " + str(false_loss_pc.shape))

    # Gather sets of points and compute their cross product to get mesh normals
    vertex_list=[1850, 1600, 2050, 5350, 5050, 5500]
    face_array = np.array([face for face in faces for vertex in vertex_list if vertex in face])
    #gt_p0 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,0]).astype(np.int32), axis=-2))(gt_pc)
    #gt_p1 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,1]).astype(np.int32), axis=-2))(gt_pc)
    #gt_p2 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,2]).astype(np.int32), axis=-2))(gt_pc)
    #print("gt_p0 shape: " + str(gt_p0.shape))
    #print("gt_p1 shape: " + str(gt_p1.shape))
    #print("gt_p2 shape: " + str(gt_p2.shape))
    #gt_vec1 = Lambda(lambda x: x[1] - x[0])([gt_p0, gt_p1])
    #gt_vec2 = Lambda(lambda x: x[1] - x[0])([gt_p0, gt_p2])
    #print("gt_vec1 shape: " + str(gt_vec1.shape))
    #print("gt_vec2 shape: " + str(gt_vec2.shape))
    #gt_normals = Lambda(lambda x: K.l2_normalize(K.tf.cross(x[0], x[1]), axis=-1), name="gt_cross_product")([gt_vec1, gt_vec2])
    #gt_normals = get_mesh_normals(gt_pc, faces, layer_name="gt_cross_product")
    gt_normals = get_mesh_normals(gt_pc, face_array, layer_name="gt_cross_product")
    print("gt_normals shape: " + str(gt_normals.shape))

    #opt_p0 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,0]).astype(np.int32), axis=-2))(optlearner_pc)
    #opt_p1 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,1]).astype(np.int32), axis=-2))(optlearner_pc)
    #opt_p2 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,2]).astype(np.int32), axis=-2))(optlearner_pc)
    #print("opt_p0 shape: " + str(opt_p0.shape))
    #print("opt_p1 shape: " + str(opt_p1.shape))
    #print("opt_p2 shape: " + str(opt_p2.shape))
    #opt_vec1 = Lambda(lambda x: x[1] - x[0])([opt_p0, opt_p1])
    #opt_vec2 = Lambda(lambda x: x[1] - x[0])([opt_p0, opt_p2])
    #print("opt_vec1 shape: " + str(opt_vec1.shape))
    #print("opt_vec2 shape: " + str(opt_vec2.shape))
    #opt_normals = Lambda(lambda x: K.l2_normalize(K.tf.cross(x[0], x[1]), axis=-1), name="opt_cross_product")([opt_vec1, opt_vec2])
    #opt_normals = get_mesh_normals(optlearner_pc, faces, layer_name="opt_cross_product")
    opt_normals = get_mesh_normals(optlearner_pc, face_array, layer_name="opt_cross_product")
    print("opt_normals shape: " + str(opt_normals.shape))
    #exit(1)

    # Learn the offset in parameters from the difference between the ground truth and learned mesh normals
    diff_normals = Lambda(lambda x: K.tf.cross(x[0], x[1]), name="diff_cross_product")([gt_normals, opt_normals])
    #pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    diff_normals_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_normals) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    print("diff_normals_NOGRAD shape: " + str(diff_normals_NOGRAD.shape))

    # Keep every 5th normal entry
    #indices = np.array([i for i in enumerate()])
    diff_normals_NOGRAD = Lambda(lambda x: x[:, ::10], name="reduce_num_normals")(diff_normals_NOGRAD)
    #exit(1)
    #index_list=[1850, 1600, 2050, 5350, 5050, 5500]
    #item_list = []
    #for id in index_list:
    #	item = Lambda(lambda x: x[:,id:(id+1), :])(pc_euclidean_diff_NOGRAD)
    #    item_list.append(item)
    #pc_euclidean_diff_NOGRAD = Concatenate(axis=-2)(item_list)
    #pc_euclidean_diff_NOGRAD = Lambda(lambda x: x[:, [1850, 1600, 2050, 5350, 5050, 5500], :])(pc_euclidean_diff_NOGRAD)
    #print("shape of output: " + str(diff_normals_NOGRAD.shape))
    #exit(1)
    #pc_euclidean_diff_NOGRAD = Flatten()(pc_euclidean_diff_NOGRAD)
    diff_normals_NOGRAD = Flatten()(diff_normals_NOGRAD)

    optlearner_architecture = Dense(2**11, activation="relu")(diff_normals_NOGRAD)
    #optlearner_architecture = Dense(2**12, activation="relu")(diff_normals_NOGRAD)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    optlearner_architecture = Dense(2**10, activation="relu")(optlearner_architecture)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    #optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    #optlearner_architecture = Dense(2**9, activation="relu")(optlearner_architecture)
    #optlearner_architecture = BatchNormalization()(optlearner_architecture)
    #optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #exit(1)
    #optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    #optlearner_architecture = Dense(1024, activation="relu")(optlearner_architecture)
    #optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    delta_d_hat = Dense(85, activation=scaled_tanh, name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape: '+str(delta_d_hat.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d, delta_d_hat])
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    #false_loss_delta_d_hat = Lambda(lambda x: K.sum(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
    false_sin_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square((x[0] - x[1]) * K.tf.sin(0.5*(x[0] - x[1]))), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_sin_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_sin_mse")(false_sin_loss_delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Lambda(lambda x: x[1]*x[0], name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat, false_loss_smpl, delta_d, delta_d_hat,delta_d_hat_NOGRAD]


def OptLearnerBothNormalsStaticSinArchitecture(param_trainable, init_wrapper, emb_size=1000):
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
    faces = smpl_params['f']    # canonical mesh faces
    print("faces shape: " + str(faces.shape))
    #exit(1)

    input_betas = Lambda(lambda x: x[:, 72:82])(optlearner_params)
    input_pose_rodrigues = Lambda(lambda x: x[:, 0:72])(optlearner_params)
    input_trans = Lambda(lambda x: x[:, 82:85])(optlearner_params)

    # Get the point cloud corresponding to these parameters
    optlearner_pc = Points3DFromSMPLParams(input_betas, input_pose_rodrigues, input_trans, smpl_params, input_info)
    print("optlearner point cloud shape: " + str(optlearner_pc.shape))
    #optlearner_pc = Lambda(lambda x: x * 0.0)[optlearner_pc]

    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_diff = Lambda(lambda x: x[0] -  x[1])([gt_pc, optlearner_pc])
    pc_euclidean_dist = Lambda(lambda x: K.sum(K.square(x),axis=-1))(pc_euclidean_diff)
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))
    #exit(1)

#    # Get the (batched) MSE between the learned and ground truth point clouds
#    false_loss_pc = Lambda(lambda x: tf.reduce_mean(tf.square(tf.subtract(x[0], x[1])), axis=[1,2]))([gt_pc, optlearner_pc])
#    false_loss_pc = Reshape(target_shape=(1,), name="pointcloud_mse")(false_loss_pc)
#    print("point cloud loss shape: " + str(false_loss_pc.shape))

    # Gather sets of points and compute their cross product to get mesh normals
    vertex_list=[1850, 1600, 2050, 5350, 5050, 5500]
    face_array = np.array([face for face in faces for vertex in vertex_list if vertex in face])
    #gt_p0 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,0]).astype(np.int32), axis=-2))(gt_pc)
    #gt_p1 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,1]).astype(np.int32), axis=-2))(gt_pc)
    #gt_p2 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,2]).astype(np.int32), axis=-2))(gt_pc)
    #print("gt_p0 shape: " + str(gt_p0.shape))
    #print("gt_p1 shape: " + str(gt_p1.shape))
    #print("gt_p2 shape: " + str(gt_p2.shape))
    #gt_vec1 = Lambda(lambda x: x[1] - x[0])([gt_p0, gt_p1])
    #gt_vec2 = Lambda(lambda x: x[1] - x[0])([gt_p0, gt_p2])
    #print("gt_vec1 shape: " + str(gt_vec1.shape))
    #print("gt_vec2 shape: " + str(gt_vec2.shape))
    #gt_normals = Lambda(lambda x: K.l2_normalize(K.tf.cross(x[0], x[1]), axis=-1), name="gt_cross_product")([gt_vec1, gt_vec2])
    #gt_normals = get_mesh_normals(gt_pc, faces, layer_name="gt_cross_product")
    gt_normals = get_mesh_normals(gt_pc, face_array, layer_name="gt_cross_product")
    print("gt_normals shape: " + str(gt_normals.shape))

    #opt_p0 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,0]).astype(np.int32), axis=-2))(optlearner_pc)
    #opt_p1 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,1]).astype(np.int32), axis=-2))(optlearner_pc)
    #opt_p2 = Lambda(lambda x: K.tf.gather(x, np.array(faces[:,2]).astype(np.int32), axis=-2))(optlearner_pc)
    #print("opt_p0 shape: " + str(opt_p0.shape))
    #print("opt_p1 shape: " + str(opt_p1.shape))
    #print("opt_p2 shape: " + str(opt_p2.shape))
    #opt_vec1 = Lambda(lambda x: x[1] - x[0])([opt_p0, opt_p1])
    #opt_vec2 = Lambda(lambda x: x[1] - x[0])([opt_p0, opt_p2])
    #print("opt_vec1 shape: " + str(opt_vec1.shape))
    #print("opt_vec2 shape: " + str(opt_vec2.shape))
    #opt_normals = Lambda(lambda x: K.l2_normalize(K.tf.cross(x[0], x[1]), axis=-1), name="opt_cross_product")([opt_vec1, opt_vec2])
    #opt_normals = get_mesh_normals(optlearner_pc, faces, layer_name="opt_cross_product")
    opt_normals = get_mesh_normals(optlearner_pc, face_array, layer_name="opt_cross_product")
    print("opt_normals shape: " + str(opt_normals.shape))
    #exit(1)

    # Learn the offset in parameters from the difference between the ground truth and learned mesh normals
    #diff_normals = Lambda(lambda x: K.tf.cross(x[0], x[1]), name="diff_cross_product")([gt_normals, opt_normals])
    diff_normals = Concatenate()([gt_normals, opt_normals])
    diff_normals_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_normals) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    print("diff_normals_NOGRAD shape: " + str(diff_normals_NOGRAD.shape))


    # Keep every 10th normal entry
    #diff_normals_NOGRAD = Lambda(lambda x: x[:, ::10], name="reduce_num_normals")(diff_normals_NOGRAD)
    diff_normals_NOGRAD = Flatten()(diff_normals_NOGRAD)

    optlearner_architecture = Dense(2**11, activation="relu")(diff_normals_NOGRAD)
    #optlearner_architecture = Dense(2**12, activation="relu")(diff_normals_NOGRAD)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    optlearner_architecture = Dense(2**10, activation="relu")(optlearner_architecture)
    optlearner_architecture = BatchNormalization()(optlearner_architecture)
    #optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    #optlearner_architecture = Dense(2**9, activation="relu")(optlearner_architecture)
    #optlearner_architecture = BatchNormalization()(optlearner_architecture)
    #optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #exit(1)
    #optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    #optlearner_architecture = Dense(1024, activation="relu")(optlearner_architecture)
    #optlearner_architecture = Dropout(0.1)(optlearner_architecture)
    delta_d_hat = Dense(85, activation=scaled_tanh, name="delta_d_hat")(optlearner_architecture)
    print('delta_d_hat shape: '+str(delta_d_hat.shape))
    #exit(1)

    # Calculate the (batched) MSE between the learned and ground truth offset in the parameters
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d, delta_d_hat])
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    #false_loss_delta_d_hat = Lambda(lambda x: K.sum(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
    false_sin_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square((x[0] - x[1]) * K.tf.sin(0.5*(x[0] - x[1]))), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_sin_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_sin_mse")(false_sin_loss_delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Lambda(lambda x: x[1]*x[0], name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat, false_loss_smpl, delta_d, delta_d_hat,delta_d_hat_NOGRAD]


def OptLearnerSilhStaticArchitecture(param_trainable, init_wrapper, emb_size=1000, dim=(256, 256)):
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
    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    pc_euclidean_diff = Lambda(lambda x: x[0] -  x[1])([gt_pc, optlearner_pc])
    pc_euclidean_dist = Lambda(lambda x: K.sum(K.square(x),axis=-1))(pc_euclidean_diff)
    print('pc euclidean dist '+str(pc_euclidean_dist.shape))
    #exit(1)
    false_loss_pc = Lambda(lambda x: K.mean(x, axis=1))(pc_euclidean_dist)
    false_loss_pc = Reshape(target_shape=(1,), name="pc_mean_euc_dist")(false_loss_pc)
    print("point cloud loss shape: " + str(false_loss_pc.shape))

    def render_silhouette(verts, dim=(256, 256), morph_mask=None):
        """ Create a(n orthographic) silhouette out of a 2D slice of the pointcloud """
        x_sf = dim[0] - 1
        y_sf = dim[1] - 1

        # Collapse the points onto the x-y plane by dropping the z-coordinate
        mesh_slice = Lambda(lambda x: x[:, :2])(verts)
        mesh_slice = Lambda(lambda x: x[:, 0] + 1)(mesh_slice)
        mesh_slice = Lambda(lambda x: x[:, 1] + 1.2)(mesh_slice)
        zoom = np.mean(dim)/2.2     # zoom into the projected pc
        mesh_slice = Lambda(lambda x: x[0] * x[1])([mesh_slice, zoom])
        coords = Lambda(lambda x: K.tf.cast(K.tf.round(x), K.tf.uint8))(mesh_slice)
        coords = Lambda(lambda x: y_sf-x[:, 1])(coords)
        coords = Lambda(lambda x: K.tf.reverse(x, axis=-1))(coords)

        # Create background to project silhouette on
        image = K.ones(shape=dim, dtype="uint8")
    #    image = Lambda(lambda x: x[0][x[1]] = 0)([image, coords])      @TODO: remove assignment

        # Define binary closing operation
        def binary_closing(image, kernel, passes):
            """ Binary closing operation with grayscale operators """
            image_shape = image.shape
            image = Reshape(target_shape=(image_shape[0], image_shape[1], 1))(image)

            kernel_shape = kernel.shape
            filter_ = Reshape(target_shape=(kernel_shape[0], kernel_shape[1], 1))(kernel)

            for i in range(passes):
                # Perform a binary closing pass
                image = Lambda(lambda x: K.tf.nn.dilation2d(x[0], x[1]))([image, kernel])
                image = Lambda(lambda x: K.tf.nn.erosion2d(x[0], x[1]))([image, kernel])


            image = Reshape(target_shape=(image_shape[0], image_shape[1]))(image)
            return image

        # Finally, perform a morphological closing operation to fill in the silhouette
        if morph_mask is None:
            # Use a circular mask as the default operator
            #inf = float("inf")
            morph_mask = np.array([[-10, -10, -10],
                                   [-10, 0.00, -10],
                                   [-10, -10, -10]
                                   ])

        closing_passes = 2
        image = Lambda(lambda x: K.tf.bitwise.invert(K.tf.cast(image, K.tf.bool)))(image)
        image = Lambda(lambda x: binary_closing(x[0], x[1], x[2]))([image, morph_mask, closing_passes])
        image = Lambda(lambda x: K.tf.cast(K.tf.invert(x), K.tf.uint8) * 255)(image)

        return image

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


