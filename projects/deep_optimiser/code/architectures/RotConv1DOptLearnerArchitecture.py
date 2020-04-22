import sys
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, Lambda, Concatenate, Dropout, BatchNormalization, Embedding, Reshape, Multiply, Add, Subtract
from keras.models import Model
from keras.activations import softsign
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.initializers import RandomUniform

sys.path.append('/data/cvfs/hjhb2/projects/mesh_rendering/code/keras_rotationnet_v2_demo_for_hidde/')
sys.path.append('/data/cvfs/ib255/shared_file_system/code/keras_rotationnet_v2/')
from points3d import Points3DFromSMPLParams, get_parameters
from smpl_np_rot_v6 import load_params
#from smpl_tf import smpl_model, smpl_model_batched
from render_mesh import Mesh
from architecture_helpers import custom_mod, init_emb_layers, false_loss, no_loss, cat_xent, mape, scaled_tanh, pos_scaled_tanh, scaled_sigmoid, centred_linear, get_mesh_normals, load_smpl_params, get_pc, get_sin_metric, get_angular_distance_metric, emb_init_weights, rot3d_from_euler, euler_from_rot3d, rot3d_from_ortho6d, ortho6d_from_rot3d, rot3d_from_rodrigues, rodrigues_from_rot3d, geodesic_loss, angle_between_vectors


def RotConv1DOptLearnerArchitecture(param_trainable, init_wrapper, smpl_params, input_info, faces, emb_size=1000, input_type="3D_POINTS"):
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
    optlearner_pc = get_pc(optlearner_params, smpl_params, input_info, faces)  # UNCOMMENT
    print("optlearner_pc shape: " + str(optlearner_pc.shape))
    #exit(1)
    #optlearner_pc = Dense(6890*3)(delta_d)
    #optlearner_pc = Reshape((6890, 3))(optlearner_pc)

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
    # In order of: right hand, right wrist, right forearm, right bicep end, right bicep, right shoulder, top of cranium, left shoulder, left bicep, left bicep end, left forearm, left wrist, left hand,
    # chest, belly/belly button, back of neck, upper back, central back, lower back/tailbone,
    # left foot, left over-ankle, left shin, left over-knee, left quadricep, left hip, right, hip, right, quadricep, right over-knee, right shin, right, over-ankle, right foot
    vertex_list = [5674, 5705, 5039, 5151, 4977, 4198, 411, 606, 1506, 1682, 1571, 2244, 2212, 3074, 3500, 460, 2878, 3014, 3021, 3365, 4606, 4588, 4671, 6877, 1799, 5262, 3479, 1187, 1102, 1120, 6740]
    #vertex_list = [5674, 5512, 5474, 5705, 6335, 5438, 5039, 5047, 5074, 5151, 5147, 5188, 4977, 4870, 4744, 4198, 4195, 5324, 411, 164, 3676, 606, 1826, 1863, 1506, 1382, 1389, 1682, 1675, 1714, 1571, 1579, 1603, 2244, 1923, 1936, 2212, 2007, 2171, 3074, 539, 4081, 3500, 6875, 3477, 460, 426, 3921, 2878, 2969, 6428, 3014, 892, 4380, 3021, 1188, 4675, 3365, 3336, 3338, 1120, 1136, 1158, 1102, 1110, 1114, 1187, 1144, 977, 3479, 872, 1161, 1799, 3150, 1804, 5262, 6567, 5268, 6877, 4359, 4647, 4671, 4630, 4462, 4588, 4641, 4600, 4606, 4622, 4644, 6740, 6744, 6737]
    #face_array = np.array([11396, 8620, 7866, 5431, 6460, 1732, 4507])
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
    diff_normals = Lambda(lambda x: K.tf.cross(x[0], x[1]), name="diff_cross_product")([gt_normals, opt_normals])
    diff_normals_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_normals) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    diff_angles = Lambda(lambda x: K.tf.subtract(x[0], x[1]), name="diff_angle")([gt_normals, opt_normals])
    diff_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_angles)
    diff_angles_norm_NOGRAD = Lambda(lambda x: K.tf.norm(x, axis=-1), name="diff_angle_norm")(diff_angles_NOGRAD)
    dist_angles = Lambda(lambda x: K.mean(K.square(x), axis=-1), name="diff_angle_mse")(diff_angles)
    dist_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(dist_angles)
    print("diff_angles shape: " + str(diff_angles.shape))
    print("dist_angles shape: " + str(dist_angles.shape))
    #pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    #print("diff_normals_NOGRAD shape: " + str(diff_normals_NOGRAD.shape))
    diff_normals_NOGRAD = Flatten()(diff_normals_NOGRAD)
    diff_angles_NOGRAD = Flatten()(diff_angles_NOGRAD)
    mesh_diff_NOGRAD = Concatenate()([diff_normals_NOGRAD, dist_angles_NOGRAD])

    if input_type == "3D_POINTS":
        optlearner_architecture = Dense(2**9, activation="relu")(vertex_diff_NOGRAD)
    if input_type == "MESH_NORMALS":
        #optlearner_architecture = Dense(2**11, activation="relu")(diff_angles_norm_NOGRAD)
        #optlearner_architecture = Dense(2**11, activation="relu")(diff_angles_NOGRAD)
        optlearner_architecture = Dense(2**9, activation="relu")(mesh_diff_NOGRAD)
    #optlearner_architecture = BatchNormalization()(optlearner_architecture)
    #optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    optlearner_architecture = Reshape((optlearner_architecture.shape[1].value, 1))(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    optlearner_architecture = Conv1D(64, 5, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Conv1D(128, 5, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Conv1D(256, 3, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Conv1D(512, 3, strides=2, activation="relu")(optlearner_architecture)
    #print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #optlearner_architecture = Flatten()(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    optlearner_architecture = Reshape((-1,))(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))

    # Learn a 6D representation of the parameters
    mapped_pose = Dense(24*6, activation="linear", name="mapped_pose")(optlearner_architecture)
    shape_params = Dense(10, activation="linear", name="shape_params")(optlearner_architecture)
    mapped_trans = Dense(6, activation="linear", name="mapped_trans")(optlearner_architecture)

    # Reshape mapped predictions into vectors
    mapped_pose_vec = Reshape((24, 3, 2), name="mapped_pose_mat")(mapped_pose)
    mapped_trans_vec = Reshape((1, 3, 2), name="mapped_trans_mat")(mapped_trans)
    print("mapped_pose shape: " + str(mapped_pose_vec.shape))
    print("mapped_trans shape: " + str(mapped_trans_vec.shape))

    # Convert 6D representation to rotation matrix in SO(3)
    rot3d_pose = rot3d_from_ortho6d(mapped_pose_vec)
    rot3d_trans = rot3d_from_ortho6d(mapped_trans_vec)
    rot3d_pose = Lambda(lambda x: x, name="rot3d_pose")(rot3d_pose)
    rot3d_trans = Lambda(lambda x: x, name="rot3d_trans")(rot3d_trans)
    print("rot3d_pose shape: " + str(rot3d_pose.shape))
    print("rot3d_trans shape: " + str(rot3d_trans.shape))

    # Cast GT difference SO(3) representation to 6D for loss calculation
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    delta_d_pose = Lambda(lambda x: x[:, 0:72])(delta_d_NOGRAD)
    delta_d_shape = Lambda(lambda x: x[:, 72:82])(delta_d_NOGRAD)
    delta_d_trans = Lambda(lambda x: x[:, 82:85])(delta_d_NOGRAD)
    print("delta_d_pose shape: " + str(delta_d_pose.shape))
    print("delta_d_shape shape: " + str(delta_d_shape.shape))
    print("delta_d_trans shape: " + str(delta_d_trans.shape))

    delta_d_pose_vec = Reshape((24, 3), name="delta_d_pose_vec")(delta_d_pose)
    delta_d_trans_vec = Reshape((1, 3), name="delta_d_trans_vec")(delta_d_trans)
    print("delta_d_pose_vec shape: " + str(delta_d_pose_vec.shape))
    print("delta_d_trans_vec shape: " + str(delta_d_trans_vec.shape))

    #rot3d_delta_d_pose = rot3d_from_euler(delta_d_pose_vec)
    #rot3d_delta_d_trans = rot3d_from_euler(delta_d_trans_vec)
    rot3d_delta_d_pose = rot3d_from_rodrigues(delta_d_pose_vec)
    rot3d_delta_d_trans = rot3d_from_rodrigues(delta_d_trans_vec)
    rot3d_delta_d_pose = Lambda(lambda x: x, name="rot3d_delta_d_pose")(rot3d_delta_d_pose)
    rot3d_delta_d_trans = Lambda(lambda x: x, name="rot3d_delta_d_trans")(rot3d_delta_d_trans)
    print("rot3d_delta_d_pose shape: " + str(rot3d_delta_d_pose.shape))
    print("rot3d_delta_d_trans shape: " + str(rot3d_delta_d_trans.shape))

    mapped_delta_d_pose_vec = ortho6d_from_rot3d(rot3d_delta_d_pose)
    mapped_delta_d_trans_vec = ortho6d_from_rot3d(rot3d_delta_d_trans)
    print("mapped_delta_d_pose_vec shape: " + str(mapped_delta_d_pose_vec.shape))
    print("mapped_delta_d_trans_vec shape: " + str(mapped_delta_d_trans_vec.shape))

    mapped_delta_d_pose = Reshape((24*3*2,), name="mapped_delta_d_pose")(mapped_delta_d_pose_vec)
    mapped_delta_d_trans = Reshape((1*3*2,), name="mapped_delta_d_trans")(mapped_delta_d_trans_vec)
    print("mapped_delta_d_pose shape: " + str(mapped_delta_d_pose.shape))
    print("mapped_delta_d_trans shape: " + str(mapped_delta_d_trans.shape))
    mapped_delta_d = Concatenate(name="mapped_delta_d")([mapped_delta_d_pose, delta_d_shape, mapped_delta_d_trans])
    print("mapped_delta_d shape: " + str(mapped_delta_d.shape))
    #exit(1)

    # Calculate L2-norm on 6D orthogonal representation or geodesic loss on SO(3) representation
    mapped_delta_d_hat = Concatenate(name="mapped_delta_d_hat")([mapped_pose, shape_params, mapped_trans])
    print("mapped_delta_d_hat shape: " + str(mapped_delta_d_hat.shape))
    #exit(1)
    mapped_delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(mapped_delta_d)

    # L2-norm loss on 6D representation
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=-1))([mapped_delta_d_NOGRAD, mapped_delta_d_hat])
    # 'Angular' loss between vectors in 6D representation
    mapped_pose_vec1 = Lambda(lambda x: x[:, :, :, 0])(mapped_pose_vec)
    mapped_pose_vec2 = Lambda(lambda x: x[:, :, :, 1])(mapped_pose_vec)
    mapped_delta_d_pose_vec1 = Lambda(lambda x: x[:, :, :, 0])(mapped_delta_d_pose_vec)
    mapped_delta_d_pose_vec2 = Lambda(lambda x: x[:, :, :, 1])(mapped_delta_d_pose_vec)
    mapped_trans_vec1 = Lambda(lambda x: x[:, :, :, 0])(mapped_trans_vec)
    mapped_trans_vec2 = Lambda(lambda x: x[:, :, :, 1])(mapped_trans_vec)
    mapped_delta_d_trans_vec1 = Lambda(lambda x: x[:, :, :, 0])(mapped_delta_d_trans_vec)
    mapped_delta_d_trans_vec2 = Lambda(lambda x: x[:, :, :, 1])(mapped_delta_d_trans_vec)
    pose_dot1 = angle_between_vectors(mapped_pose_vec1, mapped_delta_d_pose_vec1)
    pose_dot2 = angle_between_vectors(mapped_pose_vec2, mapped_delta_d_pose_vec2)
    trans_dot1 = angle_between_vectors(mapped_trans_vec1, mapped_delta_d_trans_vec1)
    trans_dot2 = angle_between_vectors(mapped_trans_vec2, mapped_delta_d_trans_vec2)
    #false_loss_delta_d_hat_shape = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=-1))([delta_d_shape, shape_params])
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(x[0] + x[1] + x[2] + x[3], axis=[-2, -1]) + x[-1])([pose_dot1, pose_dot2, trans_dot1, trans_dot2, false_loss_delta_d_hat_shape])
    # L2-norm loss on rotation matrix in SO(3)
    #false_loss_delta_d_hat_pose = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=[1, 2, 3]))([rot3d_delta_d_pose, rot3d_pose])
    #false_loss_delta_d_hat_shape = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=-1))([delta_d_shape, shape_params])
    #false_loss_delta_d_hat_trans = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=[1, 2, 3]))([rot3d_delta_d_trans, rot3d_trans])
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.tf.stack(x, axis=-1), axis=-1))([false_loss_delta_d_hat_pose, false_loss_delta_d_hat_shape, false_loss_delta_d_hat_trans])
    # Geodesic loss on rotation matrix in SO(3)
    #false_loss_delta_d_hat_pose = geodesic_loss(rot3d_delta_d_pose, rot3d_pose)
    #false_loss_delta_d_hat_shape = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=-1))([delta_d_shape, shape_params])
    #false_loss_delta_d_hat_trans = geodesic_loss(rot3d_delta_d_trans, rot3d_trans)
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.tf.stack(x, axis=-1), axis=-1))([false_loss_delta_d_hat_pose, false_loss_delta_d_hat_shape, false_loss_delta_d_hat_trans])

    # Calculate the loss for direction and magnitude separately
    #sign_loss = Lambda(lambda x: 0.5*(1. - softsign(10*x[0]*x[1])))([mapped_delta_d_NOGRAD, mapped_delta_d_hat])
    ##magnitude_loss = Lambda(lambda x: K.abs(K.abs(x[0]) - K.abs(x[1])))([delta_d_NOGRAD, delta_d_hat])
    #magnitude_loss = Lambda(lambda x: K.exp(K.abs(x[1]) - K.abs(x[0])) )([mapped_delta_d_NOGRAD, mapped_delta_d_hat])
    #weighting = 0.1
    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(x[0] + weighting*x[1], axis=-1))([sign_loss, magnitude_loss])
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
    #exit(1)

#    # Apply rotation in Euler angles
#    pos_euler_poses = euler_from_rot3d(rot3d_pose)
#    pos_euler_trans = euler_from_rot3d(rot3d_trans)
#    #pos_euler_poses = euler_from_rot3d(rot3d_delta_d_pose)     # DEBUG ONLY
#    #pos_euler_trans = euler_from_rot3d(rot3d_delta_d_trans)    # DEBUG ONLY
#    print("pos_euler_poses shape: " + str(pos_euler_poses.shape))
#    print("pos_euler_trans shape: " + str(pos_euler_trans.shape))
#
#    # Only consider first possibility for simplicity - this needs to be taken into account when evaluating predictions
#    euler_pose = Lambda(lambda x: x[:, :, :, 0], name="euler_pose")(pos_euler_poses)
#    euler_trans = Lambda(lambda x: x[:, :, :, 0], name="euler_trans")(pos_euler_trans)
#    #euler_pose = Lambda(lambda x: x[:, :, :, 1], name="euler_pose")(pos_euler_poses)
#    #euler_trans = Lambda(lambda x: x[:, :, :, 1], name="euler_trans")(pos_euler_trans)
#    print("euler_pose shape: " + str(euler_pose.shape))
#    print("euler_trans shape: " + str(euler_trans.shape))
#
#    # Reshape
#    #delta_d_hat_pose = Reshape((72,), name="delta_d_hat_pose_reshaped")(euler_pose)
#    #delta_d_hat_trans = Reshape((3,), name="delta_d_hat_trans_reshaped")(euler_trans)

    # Apply rotation in Rodrigues angles
    rodrigues_pose = rodrigues_from_rot3d(rot3d_pose)
    rodrigues_trans = rodrigues_from_rot3d(rot3d_trans)
    print("rodrigues_pose shape: " + str(rodrigues_pose.shape))
    print("rodrigues_trans shape: " + str(rodrigues_trans.shape))
    delta_d_hat_pose = Reshape((72,), name="delta_d_hat_pose_reshaped")(rodrigues_pose)
    delta_d_hat_trans = Reshape((3,), name="delta_d_hat_trans_reshaped")(rodrigues_trans)
    print("delta_d_hat_pose shape: " + str(delta_d_hat_pose.shape))
    print("delta_d_hat_trans shape: " + str(delta_d_hat_trans.shape))
    #exit(1)

    # Concatenate to form final update vector
    delta_d_hat = Concatenate(name="delta_d_hat")([delta_d_hat_pose, shape_params, delta_d_hat_trans])
    #delta_d_hat = Concatenate(name="delta_d_hat")([euler_pose, shape_params, euler_trans])
    print('delta_d_hat shape: '+str(delta_d_hat.shape))
    #exit(1)

    false_sin_loss_delta_d_hat = get_angular_distance_metric(delta_d_NOGRAD, delta_d_hat)
    #false_sin_loss_delta_d_hat = get_sin_metric(delta_d_NOGRAD, delta_d_hat)
    #false_sin_loss_delta_d_hat = get_sin_metric(delta_d_NOGRAD, delta_d_hat, average=False)
    false_sin_loss_delta_d_hat = Lambda(lambda x: x, name="delta_d_hat_sin_output")(false_sin_loss_delta_d_hat)
    #false_sin_loss_delta_d_hat = Lambda(lambda x: x, name="delta_d_hat_sin_output")(false_loss_new_pc)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Multiply(name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    # FOR DEBUGGING ONLY!!!
    #mapped_delta_d_pose_vec = Reshape((24, 3, 2))(mapped_delta_d_pose)
    #reverse_mapped_delta_d_pose = rot3d_from_ortho6d(mapped_delta_d_pose_vec)
    #rot3d_pose = Lambda(lambda x: x, name="rot3d_pose")(reverse_mapped_delta_d_pose)
    #test_rot3d_pose = Lambda(lambda x: x, name="test_rot3d_pose")(reverse_mapped_delta_d_pose)
    #rodrigues_delta_d_pose = rodrigues_from_rot3d(test_rot3d_pose)
    #rodrigues_delta_d_pose = rodrigues_from_rot3d(rot3d_delta_d_pose)
    #rodrigues_delta_d_pose = Lambda(lambda x: x, name="rodrigues_delta_d_pose")(rodrigues_delta_d_pose)

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat,  false_loss_smpl, delta_d, delta_d_hat, dist_angles, rot3d_delta_d_pose, rot3d_pose, mapped_pose, mapped_delta_d_pose]
    #return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat,  false_loss_smpl, delta_d, delta_d_hat, dist_angles, rot3d_delta_d_pose, rot3d_pose, mapped_pose, mapped_delta_d_pose, delta_d_pose_vec, rodrigues_delta_d_pose]


