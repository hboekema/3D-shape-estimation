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

def emb_init_weights(emb_params, period=None, distractor=np.pi):
    """ Embedding weights initialiser """
    def emb_init_wrapper(param, offset=False):
        def emb_init(shape):
            """ Initializer for the embedding layer """
            curr_emb_param = K.tf.gather(emb_params, param, axis=-1)
            curr_emb_param = K.tf.cast(curr_emb_param, dtype="float32")

            if offset:
                k = K.constant(distractor)
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
    y = Lambda(lambda x: K.tf.math.floormod(x - k, 2*k) - k, name=name)(x)
    return y

