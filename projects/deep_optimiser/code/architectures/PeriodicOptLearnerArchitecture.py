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
from architecture_helpers import custom_mod, init_emb_layers, false_loss, no_loss, cat_xent, mape, scaled_tanh, pos_scaled_tanh, scaled_sigmoid, centred_linear, get_mesh_normals, load_smpl_params, get_pc, get_sin_metric, get_angular_distance_metric, emb_init_weights, angle_between_vectors, split_and_reshape_euler_angles


def PeriodicOptLearnerArchitecture(param_trainable, init_wrapper, smpl_params, input_info, faces, emb_size=1000, input_type="3D_POINTS", groups=[], update_weight=1.0):
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

    # Params to train this epoch
    params_to_train = Input(shape=(85,), dtype="bool", name="params_to_train")
    print("params_to_train shape: " + str(params_to_train.shape))
    params_to_train_indices = Lambda(lambda x: x[0])(params_to_train)
    print("params_to_train_indices shape: " + str(params_to_train_indices.shape))
    params_to_train_indices = Lambda(lambda x: K.tf.where(K.tf.cast(x, "bool")))(params_to_train_indices)
    print("params_to_train_indices shape: " + str(params_to_train_indices.shape))
    params_to_train_indices = Lambda(lambda x: K.squeeze(x, 1))(params_to_train_indices)
    print("params_to_train_indices shape: " + str(params_to_train_indices.shape))
    #exit(1)

    #DROPOUT = 0.1
    DROPOUT = 0.0

    # Compute the true offset (i.e. difference) between the ground truth and learned parameters
    pi = K.constant(np.pi)
    delta_d = Lambda(lambda x: x[0] - x[1], name="delta_d")([gt_params, optlearner_params])
    #delta_d = Lambda(lambda x: x[0] - x[1], name="delta_d_no_mod")([gt_params, optlearner_params])
    #delta_d = Lambda(lambda x: K.tf.math.floormod(x - pi, 2*pi) - pi, name="delta_d")(delta_d)  # custom modulo 2pi of delta_d
    #delta_d = custom_mod(delta_d, pi, name="delta_d")  # custom modulo 2pi of delta_d
    print("delta_d shape: " + str(delta_d.shape))
    delta_d_NOGRAD = Lambda(lambda x: K.stop_gradient(x))(delta_d)
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
    dist_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(dist_angles)
    print("diff_angles shape: " + str(diff_angles.shape))
    print("dist_angles shape: " + str(dist_angles.shape))
    diff_normals_NOGRAD = Flatten()(diff_normals_NOGRAD)
    diff_angles_NOGRAD = Flatten()(diff_angles_NOGRAD)
    mesh_diff_NOGRAD = Concatenate()([diff_normals_NOGRAD, dist_angles_NOGRAD])

    # Learn to predict in conditional manner
    indices_ordering = []
    group_outputs = []
    new_params = Lambda(lambda x: x)(optlearner_params)
    group_vertex_diff_NOGRAD = Lambda(lambda x: x)(vertex_diff_NOGRAD)
    group_mesh_diff_NOGRAD = Lambda(lambda x: x)(mesh_diff_NOGRAD)
    for group in groups:
        old_params = Lambda(lambda x: x)(new_params)

        if input_type == "3D_POINTS":
            deep_opt_input = Dense(2**9, activation="relu")(group_vertex_diff_NOGRAD)
            #deep_opt_input = Dense(2**7, activation="relu")(group_vertex_diff_NOGRAD)
        if input_type == "MESH_NORMALS":
            deep_opt_input = Dense(2**9, activation="relu")(group_mesh_diff_NOGRAD)
            #deep_opt_input = Dense(2**7, activation="relu")(group_mesh_diff_NOGRAD)
        if input_type == "ONLY_NORMALS":
            deep_opt_input = Dense(2**9, activation="relu")(group_mesh_diff_NOGRAD)
            #deep_opt_input = Dense(2**7, activation="relu")(group_diff_normals_NOGRAD)

        print('deep_opt_input shape: '+str(deep_opt_input.shape))
        deep_opt_input = Reshape((-1, 1))(deep_opt_input)
        print('deep_opt_input shape: '+str(deep_opt_input.shape))

        optlearner_architecture = Conv1D(64, 5, strides=2, activation="relu")(deep_opt_input)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv1D(128, 5, strides=2, activation="relu")(optlearner_architecture)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        optlearner_architecture = Conv1D(256, 3, strides=2, activation="relu")(optlearner_architecture)
        optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        #optlearner_architecture = Conv1D(512, 3, strides=2, activation="relu")(optlearner_architecture)
        #optlearner_architecture = Dropout(DROPOUT)(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
        optlearner_architecture = Reshape((-1,))(optlearner_architecture)
        print('optlearner_architecture shape: '+str(optlearner_architecture.shape))
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

        # Get remaining indices
        rem_indices = [i for i in range(85) if i not in indices]
        group_indices_ord = indices + rem_indices
        group_ind_reord = []
        for i in sorted(group_indices_ord):
            group_ind_reord.append(group_indices_ord.index(i))

        # Apply prediction and re-render to prepare input for next group
        # Update parameters
        old_params_group = Lambda(lambda x: K.tf.gather(x, indices, axis=-1))(old_params)
        print("old_params_group shape: " + str(old_params_group.shape))
        new_params_group = Lambda(lambda x: x[0] + update_weight*x[1])([old_params_group, delta_d_hat])
        print("new_params_group shape: " + str(new_params_group.shape))
        # Complete the parameter vector
        rem_params = Lambda(lambda x: K.tf.gather(x, rem_indices, axis=-1))(old_params)
        new_params = Concatenate()([new_params_group, rem_params])
        print("new_params shape: " + str(new_params.shape))
        # Re-order the parameter vector
        new_params = Lambda(lambda x: K.tf.gather(x, group_ind_reord, axis=-1))(new_params)
        print("new_params shape: " + str(new_params.shape))

        # Render new parameters
        new_pc = Lambda(lambda x: get_pc(x, smpl_params, input_info, faces))(new_params)
        print("new_pc shape: " + str(new_pc.shape))

        # Gather input vertices
        group_pc_euclidean_diff = Lambda(lambda x: x[0] - x[1])([gt_pc, new_pc])
        group_pc_euclidean_diff_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(group_pc_euclidean_diff)
        group_vertex_diff_NOGRAD = Lambda(lambda x: K.tf.gather(x, np.array(vertex_list).astype(np.int32), axis=-2))(group_pc_euclidean_diff_NOGRAD)
        print("group_vertex_diff_NOGRAD shape: " + str(group_vertex_diff_NOGRAD.shape))
        group_vertex_diff_NOGRAD = Flatten()(group_vertex_diff_NOGRAD)
        #exit(1)
        group_gt_normals = Lambda(lambda x: get_mesh_normals(x, face_array))(gt_pc)
        print("group_gt_normals shape: " + str(group_gt_normals.shape))
        group_opt_normals = Lambda(lambda x: get_mesh_normals(x, face_array))(new_pc)
        print("group_opt_normals shape: " + str(group_opt_normals.shape))
        #exit(1)

        # Learn the offset in parameters from the difference between the ground truth and learned mesh normals
        group_diff_normals = Lambda(lambda x: K.tf.cross(x[0], x[1]))([group_gt_normals, group_opt_normals])
        group_diff_normals_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(group_diff_normals)
        group_diff_angles = Lambda(lambda x: K.tf.subtract(x[0], x[1]))([group_gt_normals, group_opt_normals])
        group_diff_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(group_diff_angles)
        group_dist_angles = Lambda(lambda x: K.mean(K.square(x), axis=-1))(group_diff_angles)
        group_dist_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(group_dist_angles)
        print("group_diff_angles shape: " + str(group_diff_angles.shape))
        print("group_dist_angles shape: " + str(group_dist_angles.shape))
        group_diff_normals_NOGRAD = Flatten()(group_diff_normals_NOGRAD)
        group_diff_angles_NOGRAD = Flatten()(group_diff_angles_NOGRAD)
        group_mesh_diff_NOGRAD = Concatenate()([group_diff_normals_NOGRAD, group_dist_angles_NOGRAD])


    if input_type == "3D_POINTS":
        deep_opt_input = Dense(2**9, activation="relu")(vertex_diff_NOGRAD)
        #deep_opt_input = Dense(2**7, activation="relu")(vertex_diff_NOGRAD)
    if input_type == "MESH_NORMALS":
        deep_opt_input = Dense(2**9, activation="relu")(mesh_diff_NOGRAD)
        #deep_opt_input = Dense(2**7, activation="relu")(mesh_diff_NOGRAD)
    if input_type == "ONLY_NORMALS":
        deep_opt_input = Dense(2**9, activation="relu")(mesh_diff_NOGRAD)
        #deep_opt_input = Dense(2**7, activation="relu")(diff_normals_NOGRAD)

    # predict shape and translation parameters
    deep_opt_input = Reshape((-1, 1))(deep_opt_input)
    print('deep_opt_input shape: '+str(deep_opt_input.shape))
    optlearner_architecture = Conv1D(64, 5, strides=2, activation="relu")(deep_opt_input)
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
    delta_d_hat = Lambda(lambda x: K.tf.gather(x, reordered_indices, axis=-1), name="delta_d_hat")(delta_d_hat)
    print("delta_d_hat shape: " + str(delta_d_hat.shape))

    #delta_d_hat_trainable = Lambda(lambda x: K.tf.boolean_mask(x[0], K.tf.cast(x[1], "bool")))([delta_d_hat, params_to_train])
    #delta_d_NOGRAD_trainable = Lambda(lambda x: K.tf.boolean_mask(x[0], K.tf.cast(x[1], "bool")))([delta_d_NOGRAD, params_to_train])
    delta_d_hat_trainable = Lambda(lambda x: K.tf.gather(x[0], K.tf.cast(x[1], "int32"), axis=-1))([delta_d_hat, params_to_train_indices])
    print("delta_d_hat_trainable shape: " + str(delta_d_hat_trainable.shape))
    delta_d_NOGRAD_trainable = Lambda(lambda x: K.tf.gather(x[0], K.tf.cast(x[1], "int32"), axis=-1))([delta_d_NOGRAD, params_to_train_indices])
    print("delta_d_NOGRAD_trainable shape: " + str(delta_d_NOGRAD_trainable.shape))

    #false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD, delta_d_hat])
    false_loss_delta_d_hat = Lambda(lambda x: K.mean(K.square(x[0] - x[1]), axis=1))([delta_d_NOGRAD_trainable, delta_d_hat_trainable])
    print("delta_d_hat loss shape: " + str(false_loss_delta_d_hat.shape))
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

    return [optlearner_input, gt_params, gt_pc, params_to_train], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat,  false_loss_smpl, delta_d, delta_d_hat, dist_angles, per_param_mse]

