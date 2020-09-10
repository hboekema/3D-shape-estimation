import sys
import numpy as np
#import tensorflow.compat.v1 as tf
import keras.backend as K
from keras.layers import Input, Dense, Flatten, Conv1D, AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D, Lambda, Concatenate, Dropout, BatchNormalization, Embedding, Reshape, Multiply, Add
from keras.activations import softsign
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
#from architecture_helpers import custom_mod, init_emb_layers, false_loss, no_loss, cat_xent, mape, scaled_tanh, pos_scaled_tanh, scaled_sigmoid, centred_linear, get_mesh_normals, load_smpl_params, get_pc, get_sin_metric, get_angular_distance_metric, emb_init_weights, angle_between_vectors, split_and_reshape_euler_angles, geodesic_loss, rot3d_from_rodrigues, rotation_loss
from architecture_helpers import custom_mod, init_emb_layers, false_loss, no_loss, cat_xent, mape, scaled_tanh, pos_scaled_tanh, scaled_sigmoid, centred_linear, get_mesh_normals, load_smpl_params, get_pc, get_sin_metric, get_angular_distance_metric, angle_between_vectors, split_and_reshape_euler_angles, geodesic_loss, rot3d_from_rodrigues, rotation_loss
#from initialisers import emb_init_weights
from initialisers.simple_initialiser_v2 import emb_init_weights


def GroupedConv1DOptLearnerArchitecture(param_trainable, init_wrapper, smpl_params, input_info, faces, emb_size=1000, input_type="3D_POINTS", groups=[]):
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

    # Get trainable parameters
    trainable_params = sorted([int(param.replace("param_", "")) for param, trainable in param_trainable.items() if trainable])
    print("trainable_params: " + str(trainable_params))
    non_trainable_params = [param for param in range(85) if param not in trainable_params]
    print("non_trainable_params: " + str(non_trainable_params))
    joints_with_trainable = sorted(np.unique([int((param - (param%3))/3) for param in trainable_params]))
    print("joints_with_trainable: " + str(joints_with_trainable))
    params_from_joints = np.concatenate([[3*i, 3*i+1, 3*i+2] for i in joints_with_trainable])
    print("params_from_joints: " + str(params_from_joints))

    #group1 = [0,1,2,3]
    #group2 = [4,5,6,7]
    #group3 = [8,9,10,11]
    #group4 = [12,13,14,15]
    #group5 = [16,17,18,19]
    #group6 = [20,21,22,23]

    #groups = [group1, group2, group3, group4, group5, group6]
    #groups = [group1 + group2, group3 + group4, group5 + group6]
    #groups = [[i] for i in range(24)]
    print("groups: " + str(groups))
    flattened_groups = [i for sublist in groups for i in sublist]
    print(flattened_groups)
    assert np.all([i in flattened_groups for i in range(24)])

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
    pc_euclidean_diff = Lambda(lambda x: x[0] - x[1])([gt_pc, optlearner_pc])
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
    vertex_list = [5674, 5705, 5039, 5151, 4977, 4198, 411, 606, 1506, 1682, 1571, 2244, 2212,
            3074, 3500, 460, 2878, 3014, 3021,
            3365, 4606, 4588, 4671, 6877, 1799, 5262, 3479, 1187, 1102, 1120, 6740]
    # with added vertices in the feet
    #vertex_list = [5674, 5705, 5039, 5151, 4977, 4198, 411, 606, 1506, 1682, 1571, 2244, 2212,
    #        3074, 3500, 460, 2878, 3014, 3021,
    #        3365, 4606, 4588, 4671, 6877, 1799, 5262, 3479, 1187, 1102, 1120, 6740, 3392, 3545, 3438, 6838, 6781, 6792]
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
    #dist_angles = Lambda(lambda x: K.mean(K.abs(x), axis=-1), name="diff_angle_mse")(diff_angles)
    dist_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(dist_angles)
    print("diff_angles shape: " + str(diff_angles.shape))
    print("dist_angles shape: " + str(dist_angles.shape))
    #pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(pc_euclidean_diff) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network
    #print("diff_normals_NOGRAD shape: " + str(diff_normals_NOGRAD.shape))
    diff_normals_NOGRAD = Flatten()(diff_normals_NOGRAD)
    diff_angles_NOGRAD = Flatten()(diff_angles_NOGRAD)
    mesh_diff_NOGRAD = Concatenate()([diff_normals_NOGRAD, dist_angles_NOGRAD])

    if input_type == "3D_POINTS":
        deep_opt_input = Dense(2**9, activation="relu")(vertex_diff_NOGRAD)
    if input_type == "MESH_NORMALS":
        deep_opt_input = Dense(2**9, activation="relu")(mesh_diff_NOGRAD)
    print('deep_opt_input shape: '+str(deep_opt_input.shape))
    deep_opt_input = Reshape((-1, 1))(deep_opt_input)
    print('deep_opt_input shape: '+str(deep_opt_input.shape))
    #DROPOUT = 0.1
    DROPOUT = 0.0
    indices_ordering = []
    group_outputs = []
    group_losses = []
    group_sin_losses = []
    group_param_mse = []
    for group in groups:
        optlearner_architecture = Conv1D(64, 5, strides=2, activation="relu")(deep_opt_input)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv1D(128, 5, strides=2, activation="relu")(optlearner_architecture)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv1D(256, 3, strides=2, activation="relu")(optlearner_architecture)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
#        optlearner_architecture = Conv1D(512, 3, strides=2, activation="relu")(optlearner_architecture)
#        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        #print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        #optlearner_architecture = Flatten()(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        optlearner_architecture = Reshape((-1,))(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        #optlearner_architecture = Dropout(0.5)(optlearner_architecture)
        #optlearner_architecture = Dense(2**7, activation="relu")(optlearner_architecture)
        #print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        #delta_d_hat = Dense(85, activation="linear", name="delta_d_hat")(optlearner_architecture)
        delta_d_hat = Dense(3*len(group), activation="linear")(optlearner_architecture)
        print('delta_d_hat shape: '+str(delta_d_hat.shape))
        group_outputs.append(delta_d_hat)
        #exit(1)

        indices = []
        for joint in group:
            j_base = 3*joint
            j1 = j_base
            j2 = j_base + 1
            j3 = j_base + 2
            indices += [j1, j2, j3]
        indices_ordering += indices

        #indices = K.constant(indices)
        ## Filter parameters such that the model is only evaluated on trainable parameters
        #delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
        #delta_d_NOGRAD_FILTERED = Lambda(lambda x: K.tf.gather(x, indices, axis=-1))(delta_d_NOGRAD)
        #delta_d_hat_FILTERED = Lambda(lambda x: K.tf.gather(x, indices, axis=-1))(delta_d_hat)

    # predict shape and translation parameters
    optlearner_architecture = Dense(2**9, activation="relu")(deep_opt_input)
    deep_opt_input = Reshape((-1, 1))(deep_opt_input)
    print('deep_opt_input shape: '+str(deep_opt_input.shape))
    optlearner_architecture = Conv1D(64, 5, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    optlearner_architecture = Conv1D(128, 5, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    optlearner_architecture = Conv1D(256, 3, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    optlearner_architecture = Conv1D(512, 3, strides=2, activation="relu")(optlearner_architecture)
    optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
    #print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #optlearner_architecture = Flatten()(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    optlearner_architecture = Reshape((-1,))(optlearner_architecture)
    print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #optlearner_architecture = Dropout(0.5)(optlearner_architecture)
    #optlearner_architecture = Dense(2**7, activation="relu")(optlearner_architecture)
    #print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
    #delta_d_hat = Dense(85, activation="linear", name="delta_d_hat")(optlearner_architecture)
    shape_params = Dense(13, activation="linear", name="shape_params")(optlearner_architecture)
    print('delta_d_hat shape: '+str(delta_d_hat.shape))
    group_outputs.append(shape_params)
    #exit(1)

    indices_ordering += [i for i in range(72, 85)]
    print(indices_ordering)

    # process indices ordering to re-order array
    reordered_indices = []
    for i in sorted(indices_ordering):
        reordered_indices.append(indices_ordering.index(i))
    reordered_indices = K.constant(reordered_indices, dtype=K.tf.int32)
    #print(reordered_indices)
    #exit(1)

    delta_d_hat = Concatenate(axis=-1)(group_outputs)
    print("delta_d_hat shape: " + str(delta_d_hat.shape))
    delta_d_hat = Lambda(lambda x: K.tf.gather(x, reordered_indices, axis=-1), name="delta_d_hat_collected")(delta_d_hat)
    print("delta_d_hat shape: " + str(delta_d_hat.shape))

    # Stop gradients for non-trainable parameters
    delta_d_hat_non_trainable = Lambda(lambda x: K.tf.gather(x, non_trainable_params, axis=-1))(delta_d_hat)
    print("delta_d_hat_non_trainable shape: " + str(delta_d_hat_non_trainable.shape))
    delta_d_hat_non_trainable_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d_hat_non_trainable)
    delta_d_hat_trainable = Lambda(lambda x: K.tf.gather(x, trainable_params, axis=-1))(delta_d_hat)
    print("delta_d_hat_trainable shape: " + str(delta_d_hat_trainable.shape))
    delta_d_hat_unordered = Concatenate()([delta_d_hat_non_trainable_NOGRAD, delta_d_hat_trainable])
    print("delta_d_hat_unordered shape: " + str(delta_d_hat_unordered.shape))

    # Re-order indices
    params_ordering = non_trainable_params + trainable_params
    delta_d_hat_ordering = []
    for i in sorted(params_ordering):
        delta_d_hat_ordering.append(params_ordering.index(i))
    delta_d_hat_ordering = K.constant(delta_d_hat_ordering, dtype=K.tf.int32)

    delta_d_hat = Lambda(lambda x: K.tf.gather(x, delta_d_hat_ordering, axis=-1), name="delta_d_hat")(delta_d_hat_unordered)
    print("delta_d_hat shape: " + str(delta_d_hat.shape))

    # Get trainable true differences
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
    delta_d_NOGRAD_trainable = Lambda(lambda x: K.tf.gather(x, trainable_params, axis=-1))(delta_d_NOGRAD)

    # Rotation loss for trainable parameters
    delta_d_hat_vec, _, _ = split_and_reshape_euler_angles(delta_d_hat)
    delta_d_NOGRAD_vec, _, _ = split_and_reshape_euler_angles(delta_d_NOGRAD)
    delta_d_hat_SO3 = rot3d_from_rodrigues(delta_d_hat_vec)
    delta_d_NOGRAD_SO3 = rot3d_from_rodrigues(delta_d_NOGRAD_vec)
    rotational_loss = rotation_loss(delta_d_NOGRAD_SO3, delta_d_hat_SO3)
    print("rotational_loss shape: " + str(rotational_loss.shape))
    rotational_loss = Lambda(lambda x: K.tf.gather(x, joints_with_trainable, axis=-1))(rotational_loss)
    print("rotational_loss shape: " + str(rotational_loss.shape))
    rotational_loss = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True), name="rotational_loss")(rotational_loss)
    print("rotational_loss shape: " + str(rotational_loss.shape))

    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD_trainable, delta_d_hat_trainable])
    #false_loss_delta_d_hat = geodesic_loss(delta_d_NOGRAD_SO3, delta_d_hat_SO3)
    #false_loss_delta_d_hat = Lambda(lambda x: x)(rotational_loss)
    #print("false_loss_delta_d_hat shape: " + str(false_loss_delta_d_hat.shape))
    #false_loss_delta_d_hat = Add()([false_loss_delta_d_hat, rotational_loss])
    print("false_loss_delta_d_hat shape: " + str(false_loss_delta_d_hat.shape))
    false_loss_delta_d_hat = Reshape(target_shape=(1,), name="delta_d_hat_mse")(false_loss_delta_d_hat)
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))

    # Metrics
    false_sin_loss_delta_d_hat = get_angular_distance_metric(delta_d_NOGRAD, delta_d_hat)
    #false_sin_loss_delta_d_hat = get_sin_metric(delta_d_NOGRAD, delta_d_hat)
    #false_sin_loss_delta_d_hat = get_sin_metric(delta_d_NOGRAD, delta_d_hat, average=False)
    false_sin_loss_delta_d_hat = Lambda(lambda x: x, name="delta_d_hat_sin_output")(false_sin_loss_delta_d_hat)
    print("delta_d_hat sin loss shape: " + str(false_sin_loss_delta_d_hat.shape))
    per_param_mse = Lambda(lambda x: K.square(K.sin(x[0] - x[1])))([delta_d_NOGRAD, delta_d_hat])
    #per_param_mse = Lambda(lambda x: K.square(x[0] - x[1]))([delta_d_NOGRAD, delta_d_hat])
    per_param_mse = Reshape((85,), name="params_mse")(per_param_mse)

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Multiply(name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    #return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat,  false_loss_smpl, delta_d, delta_d_hat, dist_angles, per_param_mse]
    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat,  false_loss_smpl, delta_d, delta_d_hat, dist_angles, per_param_mse, gt_normals, opt_normals]


