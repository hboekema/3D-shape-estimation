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


# Define custom loss functions
def trainable_param_metric(trainable_params):
    def false_param_loss(y_true, y_pred):
        return K.mean(K.tf.gather(y_pred, trainable_params, axis=1), axis=1)
    return false_param_loss

def false_loss(y_true, y_pred):
    """ Loss function that simply returns the predicted value (i.e. target is 0) """
    return y_pred

def no_loss(y_true, y_pred):
    return K.tf.reduce_sum(K.tf.multiply(y_true, 0))

def cat_xent(true_indices, y_pred):
    pred_probs = K.tf.gather_nd(y_pred, true_indices, batch_dims=1)
    print("pred_probs shape: " + str(pred_probs.shape))
    return -K.tf.math.log(pred_probs)

def mape(y_true, y_pred):
    """ Calculate the mean absolute percentage error """
    epsilon = 1e-3
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), epsilon, None))
    return 100. * K.mean(diff, axis=-1)

def SO3_loss(y_true, y_pred):
    """ Calculate geodesic loss on SO(3) rotation representations for joints, with an L2 norm loss for shape parameters """
    pass

def geodesic_loss(y_true, y_pred):
    """ Get geodesic error between two rotations, represented as rotation matrices with shape [batch_dim, no_joints, 3, 3] """
    # Find the rotation matrix that represents the difference between these matrices
    R = Lambda(lambda x: K.tf.matmul(x[0], x[1], transpose_b=True))([y_true, y_pred])
    print("R shape: " + str(R.shape))
    #exit(1)

    # Test whether the rotation matrix is valid
    #cos_theta = Lambda(lambda x: 0.5*(K.tf.trace(x) - 1))(R)
    #error = Lambda(lambda x: K.max(x, axis=-1))(cos_theta)

    # Calculate the angle this represents (tanh is to avoid argument to arccos being outside the range [-1, 1])
    theta = Lambda(lambda x: K.tf.acos(K.clip(0.5*(K.tf.trace(x) - 1), -1., 1.)))(R)
    #theta = Lambda(lambda x: K.tf.acos(K.tanh(0.5*(K.tf.trace(x) - 1))))(R)
    print("theta shape: " + str(theta.shape))
    #geodesic_error = Lambda(lambda x: K.mean(K.abs(x), axis=-1))(theta)
    geodesic_error = Lambda(lambda x: K.mean(x, axis=-1))(theta)
    print("geodesic_error shape: " + str(geodesic_error.shape))

    return geodesic_error

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
# Miscellaneous functions
def reorder_indices(indices_ordering):
    reordered_indices = []
    for i in sorted(indices_ordering):
        reordered_indices.append(indices_ordering.index(i))
    reordered_indices = K.constant(reordered_indices, dtype=K.tf.int32)
    print("reordered_indices: " + str(reordered_indices))

    return reordered_indices

def collect_and_order_outputs(group_outputs, reordered_indices):
    delta_d_hat = Concatenate(axis=-1)(group_outputs)
    delta_d_hat = Lambda(lambda x: K.tf.gather(x, reordered_indices, axis=-1))(delta_d_hat)
    print("delta_d_hat shape: " + str(delta_d_hat.shape))

    return delta_d_hat


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

def split_and_reshape_euler_angles(euler_angles):
    """ Split Euler or Rodrigues angles into vectors of pose parameters, shape parameters and translation parameters """
    euler_pose = Lambda(lambda x: x[:, 0:72])(euler_angles)
    euler_shape = Lambda(lambda x: x[:, 72:82])(euler_angles)
    euler_trans = Lambda(lambda x: x[:, 82:85])(euler_angles)

    #euler_pose = Reshape((-1, 72))(euler_pose)
    #euler_shape = Reshape((-1, 10))(euler_shape)
    #euler_trans = Reshape((-1, 3))(euler_trans)
    print("euler_pose shape: " + str(euler_pose.shape))
    print("euler_shape shape: " + str(euler_shape.shape))
    print("euler_trans shape: " + str(euler_trans.shape))

    euler_pose_vec = Reshape((24, 3))(euler_pose)
    euler_trans_vec = Reshape((1, 3))(euler_trans)
    print("euler_pose_vec shape: " + str(euler_pose_vec.shape))
    print("euler_trans_vec shape: " + str(euler_trans_vec.shape))

    return euler_pose_vec, euler_shape, euler_trans_vec


def batched_dot_product(a, b):
    """ Dot product of two vectors in last dimension of tensor (with all other dimensions being batch dimensions) """
    return Lambda(lambda x: K.sum(K.tf.multiply(x[0], x[1]), axis=-1, keepdims=True))([a, b])

def angle_between_vectors(a, b, with_norms=False):
    """ Get angle between two batched vectors """
    epsilon = 1e-5
    a_norm = Lambda(lambda x: K.tf.norm(x, axis=-1, keep_dims=True) + epsilon)(a)
    print("a_norm shape: " + str(a_norm.shape))
    b_norm = Lambda(lambda x: K.tf.norm(x, axis=-1, keep_dims=True) + epsilon)(b)
    print("b_norm shape: " + str(b_norm.shape))
    a_unit = Lambda(lambda x: x[0]/x[1])([a, a_norm])
    print("a_unit shape: " + str(a_unit.shape))
    b_unit = Lambda(lambda x: x[0]/x[1])([b, b_norm])
    print("b_unit shape: " + str(b_unit.shape))
    dot_product = batched_dot_product(a_unit, b_unit)
    print("dot_product shape: " + str(dot_product.shape))
    theta = Lambda(lambda x: K.tf.acos(K.clip(x, -1., 1.)))(dot_product)
    #theta = Lambda(lambda x: K.tf.acos(x))(dot_product)
    print("theta shape: " + str(theta.shape))

    if with_norms:
        norm_mse = Lambda(lambda x: K.square(x[0] - x[1]))([a_norm, b_norm])
        theta = Lambda(lambda x: x[0] + x[1])([theta, norm_mse])
        print("theta shape: " + str(theta.shape))
    #exit(1)
    return theta




# Metrics
def get_sin_metric(delta_d, delta_d_hat, average=True):
    """ Calculate the sin metric for evaluating different architectures """
    #false_sin_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square((x[0] - x[1]) * K.tf.sin(0.5*(x[0] - x[1]))), axis=1))([delta_d, delta_d_hat])
    false_sin_loss_delta_d_hat = Lambda(lambda x: K.square(K.tf.sin(0.5*(x[0] - x[1]))))([delta_d, delta_d_hat])
    if average:
        false_sin_loss_delta_d_hat = Lambda(lambda x: K.mean(x, axis=1))(false_sin_loss_delta_d_hat)
        false_sin_loss_delta_d_hat = Reshape(target_shape=(1,))(false_sin_loss_delta_d_hat)

    false_sin_loss_delta_d_hat = Lambda(lambda x: x, name="delta_d_hat_sin_mse")(false_sin_loss_delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))
    return false_sin_loss_delta_d_hat

def get_angular_distance_metric(delta_d, delta_d_hat, rot_form="Rodrigues"):
    """ Calculate the difference angle between two rotations """
    # Convert Euler/Rodrigues form to 3D rotation matrix form
    delta_d_pose, delta_d_shape, delta_d_trans = split_and_reshape_euler_angles(delta_d)
    delta_d_hat_pose, delta_d_hat_shape, delta_d_hat_trans = split_and_reshape_euler_angles(delta_d_hat)

    delta_d_rots = Concatenate(axis=-2)([delta_d_pose, delta_d_trans])
    delta_d_hat_rots = Concatenate(axis=-2)([delta_d_hat_pose, delta_d_hat_trans])

    if rot_form == "Rodrigues":
        rot3d_delta_d = rot3d_from_rodrigues(delta_d_rots)
        rot3d_delta_d_hat = rot3d_from_rodrigues(delta_d_hat_rots)
    elif rot_form == "Euler":
        rot3d_delta_d = rot3d_from_euler(delta_d_rots)
        rot3d_delta_d_hat = rot3d_from_euler(delta_d_hat_rots)
    else:
        raise ValueError("Argument to rot_form '{}' not recognised (should be either 'Rodrigues' or 'Euler').".format(rot_form))

    #dot_product = Lambda(lambda x: K.tf.reduce_sum(K.tf.multiply(x[:, :, 0], x[:, :, 1]), axis=-1))(rot3d_delta_d_hat)
    #error = Lambda(lambda x: K.mean(x, axis=1))(dot_product)

    # Find the rotation matrix that represents the difference between these matrices
    R = Lambda(lambda x: K.tf.matmul(x[0], x[1], transpose_b=True))([rot3d_delta_d, rot3d_delta_d_hat])
    #R = Lambda(lambda x: K.tf.matmul(x[1], x[0], transpose_b=True))([rot3d_delta_d, rot3d_delta_d_hat])
    print("R shape: " + str(R.shape))
    #exit(1)

    # Test whether the rotation matrix is valid
    #cos_theta = Lambda(lambda x: 0.5*(K.tf.trace(x) - 1))(R)
    #error = Lambda(lambda x: K.max(x, axis=-1))(cos_theta)

    # Calculate the angle this represents
    theta = Lambda(lambda x: K.tf.acos(K.clip(0.5*(K.tf.trace(x) - 1), -1., 1.)))(R)
    print("theta shape: " + str(theta.shape))
    #sin_theta = Lambda(lambda x: K.sin(x))(theta)
    #print("sin_theta shape: " + str(sin_theta.shape))
    # Can normalise the error to lie in range [0, 1]
    angular_error = Lambda(lambda x: K.mean(x, axis=-1))(theta)
    #angular_error = Lambda(lambda x: K.mean(K.abs(x), axis=-1))(theta)
    #angular_error = Lambda(lambda x: K.mean(K.abs(x)/np.pi, axis=-1))(theta)
    #angular_error = Lambda(lambda x: K.mean(K.abs(x), axis=-1))(sin_theta)

    # Calculate the MSE error of the shape predictions
    shape_error = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=-1))([delta_d_shape, delta_d_hat_shape])

    # Combine the angular MAE with the shape MSE
    error = Add(name="delta_d_hat_sin_mse_unshaped")([angular_error, shape_error])
    #error = Lambda(lambda x: x, name="delta_d_hat_sin_mse_unshaped")(angular_error)
    error = Reshape((1,), name="delta_d_hat_sin_mse")(error)

    return error


# Initialisers
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

def emb_init_weights(emb_params, period=None, distractor=np.pi, pose_offset={}):
    """ Embedding weights initialiser """
    def emb_init_wrapper(param, offset=False):
        def emb_init(shape):
            """ Initializer for the embedding layer """
            curr_emb_param = K.tf.gather(emb_params, param, axis=-1)
            curr_emb_param = K.tf.cast(curr_emb_param, dtype="float32")

            if offset or "param_{:02d}".format(param) in pose_offset.keys():
                if offset:
                    k = K.constant(distractor["param_{:02d}".format(param)])
                else:
                    k = K.constant(pose_offset["param_{:02d}".format(param)])
                offset_value = K.random_uniform(shape=[shape[0]], minval=-k, maxval=k, dtype="float32")
                #offset_value = K.random_uniform(shape=[shape[0]], minval=k, maxval=k, dtype="float32")
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
    #y = Lambda(lambda x: K.tf.math.floormod(x - k, 2*k) - k, name=name)(x)
    y = Lambda(lambda x: K.tf.floormod(x - k, 2*k) - k)(x)
    return y


# Normal distribution helper functions
def normal_sample(mu, sigma):
    """ Sample from a univariate normal distribution in a batched manner """
    return Lambda(lambda x: x[0] + x[1]*K.random_normal(K.tf.shape(x[1])))([mu, sigma])

def normal_log_prob_pos(sample, mu, sigma):
    """ Return the (strictly positive) log probability of a continuous normal random sample """
    # Not computationally efficient, but the addition of 1 is necessary in the log-likelihood for this distribution to limit it to [0, inf)
    sample_mu_squared = Lambda(lambda x: K.square(x[0] - x[1]))([sample, mu])
    sample_prob = Lambda(lambda x: K.exp(-x[0]/(2*K.square(x[1]))) / (x[1] * np.sqrt(2*np.pi)))([sample_mu_squared, sigma])
    return Lambda(lambda x: K.log(1 + x))(sample_prob)

def normal_log_prob(sample, mu, sigma):
    """ Return the log probability of a continuous normal random sample """
    sample_mu_squared = Lambda(lambda x: K.square(x[0] - x[1]))([sample, mu])
    sigma_squared = Lambda(lambda x: K.square(x))(sigma)

    log_prob = Lambda(lambda x: -0.5*(np.log(2*np.pi) - K.log(x[1]) - x[0]/x[1]))([sample_mu_squared, sigma_squared])
    return log_prob

def normal_entropy_pos(sigma):
    """ Return the (strictly positive) differential entropy of the normal distribution with a std dev of sigma """
    return Lambda(lambda x: K.log(1 + (x * np.sqrt(2*np.pi*np.exp(1)))))(sigma)

def normal_entropy(sigma):
    """ Return the differential entropy of the normal distribution with a std dev of sigma """
    #return Lambda(lambda x: K.log(1 + (x * np.sqrt(2*np.pi*np.exp(1)))))(sigma)
    return Lambda(lambda x: K.log(x * np.sqrt(2*np.pi*np.exp(1))))(sigma)

def distributional_model_loss(eligibility, entropy, weighting=0.01):
    """ Calculate the loss function for the probabilistic model - this is a trade-off between high probability of accurate predictions and precision (or how certain the network is) """
    return Lambda(lambda x: -(x[0] + weighting*x[1]), name="dist_goodness")([eligibility, entropy])

