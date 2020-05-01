import sys
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
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
from architecture_helpers import custom_mod, init_emb_layers, false_loss, no_loss, cat_xent, mape, scaled_tanh, pos_scaled_tanh, scaled_sigmoid, centred_linear, get_mesh_normals, load_smpl_params, get_pc, get_sin_metric, emb_init_weights, angle_between_vectors, split_and_reshape_euler_angles
from inputs import standard_input
from diff_helpers import get_delta_d, get_delta_d_loss, get_pc_loss
from geometry import get_vertices, get_normals_from_vertices


def GroupedConv1DOptLearnerArchitecture(param_trainable, init_wrapper, smpl_params, input_info, faces, emb_size=1000, input_type="3D_POINTS", groups=[], DROPOUT=0.0):
    """ Optimised learner network architecture """
    # Get network inputs
    optlearner_params, gt_params, gt_pc = standard_input(emb_size, param_trainable, init_wrapper)

    # Assert all pose parameters are specified in group hierarchy
    assert np.all([joint in np.flatten(groups) for joint in range(0, 24)])

    # Compute the true offset (i.e. difference) between the ground truth and learned parameters
    delta_d = get_delta_d(gt_params, optlearner_params)

    # Calculate the (batched) MSE between the learned parameters and the ground truth parameters
    false_loss_delta_d = get_delta_d_loss(delta_d)

    # Load SMPL model and get necessary parameters
    optlearner_pc = get_pc(optlearner_params, smpl_params, input_info, faces)
    print("optlearner_pc shape: " + str(optlearner_pc.shape))

    # Get the (batched) Euclidean loss between the learned and ground truth point clouds
    false_loss_pc = get_pc_loss(gt_pc, optlearner_pc)

    # Gather sets of points and compute their cross product to get mesh normals
    vertex_list = get_vertices()

    vertex_diff_NOGRAD = get_vertex_diff(gt_pc, optlearner_pc, vertex_list)
    gt_normals, opt_normals = get_normals_from_vertices(gt_pc, optlearner_pc, vertex_list)

    diff_normals = get_normals_diff(gt_normals, opt_normals)
    diff_normals_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_normals) # This is added to avoid influencing embedding layer parameters by a "bad" gradient network

    diff_angles, dist_angles = get_normals_angle_diff(gt_normals, opt_normals)
    diff_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(diff_angles)
    dist_angles_NOGRAD =  Lambda(lambda x: K.stop_gradient(x))(dist_angles)

    # Learn the offset in parameters from the difference between the ground truth and learned mesh normals
    diff_normals_NOGRAD = Flatten()(diff_normals_NOGRAD)
    diff_angles_NOGRAD = Flatten()(diff_angles_NOGRAD)
    mesh_diff_NOGRAD = Concatenate()([diff_normals_NOGRAD, dist_angles_NOGRAD])

    if input_type == "3D_POINTS":
        deep_opt_input = Dense(2**9, activation="relu")(vertex_diff_NOGRAD)
    if input_type == "MESH_NORMALS":
        deep_opt_input = Dense(2**9, activation="relu")(diff_normals_NOGRAD)
    if input_type == "COMBINED":
        deep_opt_input = Dense(2**9, activation="relu")(mesh_diff_NOGRAD)
    print('deep_opt_input shape: '+str(deep_opt_input.shape))

    # Predict pose in the specified groups
    if output_type == "6D":
        indices_ordering, group_outputs = grouped_6D_network(deep_opt_input, groups, DROPOUT)

        # Get delta_d pose parameters in orthogonal 6D representation
        delta_d_pose = delta_d_to_ortho6d(delta_d_NOGRAD, indices_ordering)
        # Get delta_d_hat pose parameters
        delta_d_hat_pose = Concatenate(axis=-1)(group_outputs)

        # Get loss for pose parameters
        false_loss_delta_d_hat = delta_d_hat_mse(delta_d_pose, delta_d_hat_pose)

        # Convert delta_d_hat to Rodrigues form
        delta_d_hat_rod_pose = ortho6d_pose_to_rodrigues(delta_d_hat_pose)

        # Predict shape and translation parameters
        shape_params = shape_network(deep_opt_input, DROPOUT)
        indices_ordering += [i for i in range(72, 85)]
        print("indices_ordering: " + strindices_ordering))

        # Get loss on shape and translation parameters


        # Process indices ordering to re-order array
        reordered_indices = reorder_indices(indices_ordering)

        # Re-order predictions
        delta_d_hat = collect_and_order_outputs(group_outputs, reordered_indices)



    elif output_type == "Rodrigues":
        indices_ordering, group_outputs = grouped_network(deep_opt_input, groups, DROPOUT)

        # Predict shape and translation parameters
        shape_params = shape_network(deep_opt_input, DROPOUT)
        group_outputs.append(shape_params)
        indices_ordering += [i for i in range(72, 85)]
        print("indices_ordering: " + strindices_ordering))

        # Process indices ordering to re-order array
        reordered_indices = reorder_indices(indices_ordering)

        # Re-order predictions
        delta_d_hat = collect_and_order_outputs(group_outputs, reordered_indices)

        # Calculate loss on predictions
        false_loss_delta_d_hat = delta_d_hat_mse(delta_d_NOGRAD, delta_d_hat)
    else:
        print("Output type '{}' not recognised. Exiting.".format(output_type))
        exit(1)


    # Metrics
    #false_sin_loss_delta_d_hat = paramwise_sin_metric(delta_d_NOGRAD, delta_d_hat)
    false_sin_loss_delta_d_hat = get_angular_distance_metric(delta_d_NOGRAD, delta_d_hat)
    per_param_mse = param_metric(delta_d_NOGRAD, delta_d_hat, sine_metric=True)

    # Prevent model from using the delta_d_hat gradient in final loss
    delta_d_hat_NOGRAD = Lambda(lambda x: K.stop_gradient(x), name='optlearner_output_NOGRAD')(delta_d_hat)

    # False loss designed to pass the learned offset as a gradient to the embedding layer
    false_loss_smpl = Multiply(name="smpl_diff")([optlearner_params, delta_d_hat_NOGRAD])
    print("smpl loss shape: " + str(false_loss_smpl.shape))

    return [optlearner_input, gt_params, gt_pc], [optlearner_params, false_loss_delta_d, optlearner_pc, false_loss_pc, false_loss_delta_d_hat, false_sin_loss_delta_d_hat,  false_loss_smpl, delta_d, delta_d_hat, dist_angles, per_param_mse]


