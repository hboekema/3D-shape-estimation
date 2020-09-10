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
    return K.mean(y_pred)    # @TODO: ascertain that K.mean() does not change results

def false_loss_summed(y_true, y_pred):
    """ Loss function that simply returns the predicted value (i.e. target is 0) """
    return K.sum(y_pred)    # @TODO: ascertain that K.sum() does not change results

def no_loss(y_true, y_pred):
    return K.sum(y_true*0.0)

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

def rotation_loss(y_true, y_pred):
    """ Rotate a unit vector by 3D rotation matrices y_true and y_pred (of shape [batch_dim, num_joints, 3, 3]) and take MSE over the resulting vectors """
    unit_vector = [[0., 0., 1.]]
    unit_vector = K.constant(unit_vector, dtype="float")
    print("unit_vector shape: " + str(unit_vector.shape))
    unit_vector = Lambda(lambda x, unit_vector: K.tf.tile(unit_vector, [K.tf.shape(x)[0], 24]), arguments={'unit_vector': unit_vector})(y_true)
    print("unit_vector shape: " + str(unit_vector.shape))
    unit_vector = Reshape((24, 3, 1))(unit_vector)
    print("unit_vector shape: " + str(unit_vector.shape))

    print("y_true shape: " + str(y_true.shape))
    print("y_pred shape: " + str(y_pred.shape))

    true_vector = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[3, 2]))([y_true, unit_vector])
    print("true_vector shape: " + str(true_vector.shape))
    true_vector = Reshape((-1, 3))(true_vector)
    print("true_vector shape: " + str(true_vector.shape))

    pred_vector = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[3, 2]))([y_pred, unit_vector])
    print("pred_vector shape: " + str(pred_vector.shape))
    pred_vector = Reshape((-1, 3))(pred_vector)
    print("pred_vector shape: " + str(pred_vector.shape))


    rotational_error = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=-1))([true_vector, pred_vector])
    print("rotational_error shape: " + str(rotational_error.shape))

    return rotational_error

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


# Projection helpers
def rot3d_from_ortho6d(ortho6d, gram_schmidt=True):
    a1 = Lambda(lambda x: x[:, :, :, 0])(ortho6d)
    a2 = Lambda(lambda x: x[:, :, :, 1])(ortho6d)
    print("a1 shape: " + str(a1.shape))
    print("a2 shape: " + str(a2.shape))

    b1 = Lambda(lambda x: K.tf.nn.l2_normalize(x, dim=-1))(a1)
    print("b1 shape: " + str(b1.shape))
    if gram_schmidt:
        # Use Gram-Schmidt process to find orthogonal vectors
        b1_projected = Lambda(lambda x: K.tf.reduce_sum(K.tf.multiply(x[1], x[0]), keep_dims=True) * x[1])([a2, b1])
        b2 = Lambda(lambda x: K.tf.nn.l2_normalize(x[0] - x[1], dim=-1))([a2, b1_projected])
        print("b2 shape: " + str(b2.shape))
        b3 = Lambda(lambda x: K.tf.cross(x[0], x[1]))([b1, b2])
        print("b3 shape: " + str(b3.shape))
    else:
        # Use cross products to find orthogonal vectors in R^3
        b3 = Lambda(lambda x: K.tf.nn.l2_normalize(K.tf.cross(x[0], x[1]), dim=-1))([b1, a2])
        print("b3 shape: " + str(b3.shape))
        b2 = Lambda(lambda x: K.tf.cross(x[0], x[1]))([b1, b3])
        print("b2 shape: " + str(b2.shape))

    R = Lambda(lambda x: K.stack(x, axis=-1))([b1, b2, b3])
    print("R shape: " + str(R.shape))
    #R = Reshape((-1, 3, 3))(R)
    #print("R shape: " + str(R.shape))
    #exit(1)

    return R

def ortho6d_from_rot3d(rot3d):
    ortho6d = Lambda(lambda x: x[:, :, :, 0:2])(rot3d)
    print("ortho_6d shape: " + str(ortho6d.shape))
    #exit(1)
    return ortho6d

def rot3d_from_rodrigues(rodrigues):
    # Convert axis-angle vector of shape [batch_dim, num_joints, 3] to 3D rotation matrix of shape [batch_dim, num_joints, 3, 3]
    # get the angle of rotation and unit vector
    epsilon = 1e-8
    theta = Lambda(lambda x: K.tf.norm(x, axis=-1, keep_dims=True) + epsilon)(rodrigues)
    print("theta shape: " + str(theta.shape))
    unit_r = Lambda(lambda x: x[0]/x[1])([rodrigues, theta])
    print("unit_r shape: " + str(unit_r.shape))

    # Get x-, y-, and z- components of unit axis vector
    r_x = Lambda(lambda x: K.expand_dims(x[:, :, 0]))(unit_r)
    r_y = Lambda(lambda x: K.expand_dims(x[:, :, 1]))(unit_r)
    r_z = Lambda(lambda x: K.expand_dims(x[:, :, 2]))(unit_r)
    print("r_x shape: " + str(r_x.shape))

    r_xx = Lambda(lambda x: x[0]*x[1])([r_x, r_x])
    r_yy = Lambda(lambda x: x[0]*x[1])([r_y, r_y])
    r_zz = Lambda(lambda x: x[0]*x[1])([r_z, r_z])
    r_xy = Lambda(lambda x: x[0]*x[1])([r_x, r_y])
    r_xz = Lambda(lambda x: x[0]*x[1])([r_x, r_z])
    r_yz = Lambda(lambda x: x[0]*x[1])([r_y, r_z])
    print("r_xx shape: " + str(r_xx.shape))

    c = Lambda(lambda x: K.cos(x))(theta)
    s = Lambda(lambda x: K.sin(x))(theta)
    C = Lambda(lambda x: 1-x)(c)

    r_xs = Lambda(lambda x: x[0]*x[1])([r_x, s])
    r_ys = Lambda(lambda x: x[0]*x[1])([r_y, s])
    r_zs = Lambda(lambda x: x[0]*x[1])([r_z, s])

    # Construct rotation matrix
    R11 = Lambda(lambda x: x[0]*x[1] + x[2])([r_xx, C, c])
    R12 = Lambda(lambda x: x[0]*x[1] - x[2])([r_xy, C, r_zs])
    R13 = Lambda(lambda x: x[0]*x[1] + x[2])([r_xz, C, r_ys])
    R21 = Lambda(lambda x: x[0]*x[1] + x[2])([r_xy, C, r_zs])
    R22 = Lambda(lambda x: x[0]*x[1] + x[2])([r_yy, C, c])
    R23 = Lambda(lambda x: x[0]*x[1] - x[2])([r_yz, C, r_xs])
    R31 = Lambda(lambda x: x[0]*x[1] - x[2])([r_xz, C, r_ys])
    R32 = Lambda(lambda x: x[0]*x[1] + x[2])([r_yz, C, r_xs])
    R33 = Lambda(lambda x: x[0]*x[1] + x[2])([r_zz, C, c])
    print("R11 shape: " + str(R11.shape))

    R = Concatenate()([R11, R12, R13, R21, R22, R23, R31, R32, R33])
    print("R shape: " + str(R.shape))
    R = Reshape((-1, 3, 3))(R)
    print("R shape: " + str(R.shape))

    return R

def rodrigues_from_rot3d(rot3d):
    # Convert 3D rotation matrix to (positive) axis-angle vector
    # Gather elements of the rotation matrix
    R11 = Lambda(lambda x: K.expand_dims(x[:, :, 0, 0]))(rot3d)
    R12 = Lambda(lambda x: K.expand_dims(x[:, :, 0, 1]))(rot3d)
    R13 = Lambda(lambda x: K.expand_dims(x[:, :, 0, 2]))(rot3d)
    R21 = Lambda(lambda x: K.expand_dims(x[:, :, 1, 0]))(rot3d)
    R22 = Lambda(lambda x: K.expand_dims(x[:, :, 1, 1]))(rot3d)
    R23 = Lambda(lambda x: K.expand_dims(x[:, :, 1, 2]))(rot3d)
    R31 = Lambda(lambda x: K.expand_dims(x[:, :, 2, 0]))(rot3d)
    R32 = Lambda(lambda x: K.expand_dims(x[:, :, 2, 1]))(rot3d)
    R33 = Lambda(lambda x: K.expand_dims(x[:, :, 2, 2]))(rot3d)
    print("R11 shape: " + str(R11.shape))

    # Calculate un-normalised angle vector
    r_x = Lambda(lambda x: x[0] - x[1])([R32, R23])
    r_y = Lambda(lambda x: x[0] - x[1])([R13, R31])
    r_z = Lambda(lambda x: x[0] - x[1])([R21, R12])
    print("r_x shape: " + str(r_x.shape))

    r_xyz = Concatenate()([r_x, r_y, r_z])
    print("r_xyz shape: " + str(r_xyz.shape))
    epsilon = 1e-8
    r_magnitude = Lambda(lambda x: K.tf.norm(x, axis=-1, keep_dims=True) + epsilon)(r_xyz)
    print("r_magnitude shape: " + str(r_magnitude.shape))

    # Calculate angle
    trace = Lambda(lambda x: x[0] + x[1] + x[2])([R11, R22, R33])
    theta = Lambda(lambda x: K.tf.atan2(x[0], x[1] - 1))([r_magnitude, trace])
    print("theta shape: " +str(theta.shape))

    # Form compressed vector and return this
    r_unit = Lambda(lambda x: x[0]/x[1])([r_xyz, r_magnitude])
    r = Lambda(lambda x: x[0]*x[1])([theta, r_unit])
    print("r_unit shape: " + str(r_unit.shape))
    print("r shape: " + str(r.shape))

    return r

def rot3d_from_euler(euler):
    # Convert Euler-Rodrigues angles to a 3D rotation matrix (with order of rotation X-Y-Z)
    euler_x = Lambda(lambda x: x[:, :, 0])(euler)
    euler_y = Lambda(lambda x: x[:, :, 1])(euler)
    euler_z = Lambda(lambda x: x[:, :, 2])(euler)

    cx = Lambda(lambda x: K.cos(x))(euler_x)
    sx = Lambda(lambda x: K.sin(x))(euler_x)
    cy = Lambda(lambda x: K.cos(x))(euler_y)
    sy = Lambda(lambda x: K.sin(x))(euler_y)
    cz = Lambda(lambda x: K.cos(x))(euler_z)
    sz = Lambda(lambda x: K.sin(x))(euler_z)

    R11 = Lambda(lambda x: x[0]*x[1])([cy, cz])
    R12 = Lambda(lambda x: x[0]*x[1]*x[2] - x[3]*x[4])([sx, sy, cz, cx, sz])
    R13 = Lambda(lambda x: x[0]*x[1]*x[2] + x[3]*x[4])([cx, sy, cz, sx, sz])
    R21 = Lambda(lambda x: x[0]*x[1])([cy, sz])
    R22 = Lambda(lambda x: x[0]*x[1]*x[2] + x[3]*x[4])([sx, sy, sz, cx, cz])
    R23 = Lambda(lambda x: x[0]*x[1]*x[2] - x[3]*x[4])([cx, sy, sz, sx, cz])
    R31 = Lambda(lambda x: -x)(sy)
    R32 = Lambda(lambda x: x[0]*x[1])([sx, cy])
    R33 = Lambda(lambda x: x[0]*x[1])([cx, cy])
    print("R11 shape: " + str(R11.shape))

    R = Lambda(lambda x: K.stack(x, axis=-1))([R11, R12, R13, R21, R22, R23, R31, R32, R33])
    print("R shape: " + str(R.shape))
    R = Reshape((-1, 3, 3))(R)
    print("R shape: " + str(R.shape))
    #exit(1)

    return R

def euler_from_rot3d(rot3d):
    # Rotation matrix in SO(3) to Euler-Rodrigues form. Note that many-to-one mappings exist, so all possible forms within [-pi, pi] are returned
    # Gather salient rotation matrix entries
    R11 = Lambda(lambda x: x[:, :, 0, 0])(rot3d)
    R12 = Lambda(lambda x: x[:, :, 0, 1])(rot3d)
    R13 = Lambda(lambda x: x[:, :, 0, 2])(rot3d)
    R21 = Lambda(lambda x: x[:, :, 1, 0])(rot3d)
    R31 = Lambda(lambda x: x[:, :, 2, 0])(rot3d)
    R32 = Lambda(lambda x: x[:, :, 2, 1])(rot3d)
    R33 = Lambda(lambda x: x[:, :, 2, 2])(rot3d)

    # Calculate y-axis rotation
    pi = K.constant(np.pi)
    def normal_case(R11, R21, R31, R32, R33):
        y1 = Lambda(lambda x: -K.tf.asin(x))(R31)
        y2 = Lambda(lambda x: pi - x)(y1)

        # Calculate cosine of possible y-axis rotations
        cy1 = Lambda(lambda x: K.cos(x))(y1)
        cy2 = Lambda(lambda x: K.cos(x))(y2)

        # Sign-correct the remaining rotation terms
        R11_cy1 = Lambda(lambda x: x[0]/x[1])([R11, cy1])
        R11_cy2 = Lambda(lambda x: x[0]/x[1])([R11, cy2])
        R21_cy1 = Lambda(lambda x: x[0]/x[1])([R21, cy1])
        R21_cy2 = Lambda(lambda x: x[0]/x[1])([R21, cy2])
        R32_cy1 = Lambda(lambda x: x[0]/x[1])([R32, cy1])
        R32_cy2 = Lambda(lambda x: x[0]/x[1])([R32, cy2])
        R33_cy1 = Lambda(lambda x: x[0]/x[1])([R33, cy1])
        R33_cy2 = Lambda(lambda x: x[0]/x[1])([R33, cy2])

        # Find possible x- and z-axis rotations
        x1 = Lambda(lambda x: K.tf.atan2(x[0], x[1]))([R32_cy1, R33_cy1])
        x2 = Lambda(lambda x: K.tf.atan2(x[0], x[1]))([R32_cy2, R33_cy2])
        z1 = Lambda(lambda x: K.tf.atan2(x[0], x[1]))([R21_cy1, R11_cy1])
        z2 = Lambda(lambda x: K.tf.atan2(x[0], x[1]))([R21_cy2, R11_cy2])
        print("x1 shape: " + str(x1.shape))

        # Gather corresponding elements
        #euler1 = Concatenate()([x1, y1, z1])
        euler1 = Lambda(lambda x: K.stack(x, axis=-1))([x1, y1, z1])
        print("euler1 shape: " + str(euler1.shape))
        #euler1 = Reshape((-1, 3))(euler1)
        #print("euler1 shape: " + str(euler1.shape))
        #euler2 = Concatenate()([x2, y2, z2])
        euler2 = Lambda(lambda x: K.stack(x, axis=-1))([x2, y2, z2])
        print("euler2 shape: " + str(euler2.shape))
        #euler2 = Reshape((-1, 3))(euler2)
        #print("euler2 shape: " + str(euler2.shape))
        euler = Lambda(lambda x: K.stack(x, axis=-1))([euler1, euler2])
        print("euler shape: " + str(euler.shape))
        #euler = Reshape((-1, 3, 2))(euler)
        #print("euler shape: " + str(euler.shape))
        #exit(1)
        return euler

    euler = normal_case(R11, R21, R31, R32, R33)
    return euler


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

def get_angular_distance_metric(delta_d, delta_d_hat, rot_form="Rodrigues", include_shape=True):
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

    # Get the geodesic loss between these matrices
    angular_error = geodesic_loss(rot3d_delta_d, rot3d_delta_d_hat)

    if include_shape:
        # Calculate the MSE error of the shape predictions
        shape_error = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=-1))([delta_d_shape, delta_d_hat_shape])

        # Combine the angular MAE with the shape MSE
        error = Add(name="delta_d_hat_sin_mse_unshaped")([angular_error, shape_error])
    else:
        error = Lambda(lambda x: x, name="delta_d_hat_sin_mse_unshaped")(angular_error)
    error = Reshape((1,), name="delta_d_hat_sin_mse")(error)

    return error


# Initialisers
def init_emb_layers(index, emb_size, param_trainable, init_wrapper):
        """ Initialise the parameter embedding layers """
        emb_layers = []
        num_params = 85
        blacklist = range(72, 85)
        for i in range(num_params):
            layer_name = "param_{:02d}".format(i)
            initialiser = init_wrapper(parameter=i)

            if i in blacklist:
                trainable = False
            else:
                trainable = param_trainable[layer_name]
            #emb_layer = Embedding(emb_size, 1, name=layer_name, trainable=param_trainable[layer_name], embeddings_initializer=initialiser)(index)
            emb_layer = Embedding(emb_size, 1, name=layer_name, trainable=trainable, embeddings_initializer=initialiser)(index)
            emb_layers.append(emb_layer)

        return emb_layers

#def emb_init_weights(emb_params, period=None, distractor=np.pi, pose_offset={}, dist="uniform", reset_to_zero=False):
#    """ Embedding weights initialiser """
#    recognised_modes = ["uniform", "gaussian", "normal"]
#    assert dist in recognised_modes
#
#    def emb_init_wrapper(param, offset=False):
#        def emb_init(shape):
#            """ Initializer for the embedding layer """
#            curr_emb_param = K.tf.gather(emb_params, param, axis=-1)
#            curr_emb_param = K.tf.cast(curr_emb_param, dtype="float32")
#            epsilon = 1e-5
#
#            if reset_to_zero:
#                #print(emb_params.shape)
#                epsilon = K.constant(np.tile(epsilon, shape))
#                #print(epsilon.shape)
#                curr_emb_param = K.zeros_like(curr_emb_param)
#                #print(curr_emb_param.shape)
#                curr_emb_param = Add()([curr_emb_param, epsilon])
#                #print(curr_emb_param.shape)
#                #exit(1)
#            elif offset or "param_{:02d}".format(param) in pose_offset.keys():
#                if offset:
#                    k = K.constant(distractor["param_{:02d}".format(param)])
#                else:
#                    k = K.constant(pose_offset["param_{:02d}".format(param)])
#
#                if dist == "uniform":
#                    offset_value = K.random_uniform(shape=[shape[0]], minval=-k, maxval=k, dtype="float32")
#                elif dist == "normal" or dist == "gaussian":
#                    offset_value = K.random_normal(shape=[shape[0]], mean=0.0, stddev=k, dtype="float32")
#
#                #print(offset_value)
#                #exit(1)
#                if period is not None and shape[0] % period == 0:
#                    block_size = shape[0] // period
#                    #factors = Concatenate()([K.random_normal(shape=[block_size], mean=np.sqrt((i+1)/period), stddev=0.01) for i in range(period)])
#                    factors = Concatenate()([K.random_normal(shape=[block_size], mean=np.sqrt(float(i+1)/period), stddev=0.01) for i in range(period)])
#                    offset_value = Multiply()([offset_value, factors])
#
#                curr_emb_param = Add()([curr_emb_param, offset_value])
#
#            init = Reshape(target_shape=[shape[1]])(curr_emb_param)
#            #print("init shape: " + str(init.shape))
#            #exit(1)
#            return init
#        return emb_init
#    return emb_init_wrapper

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

