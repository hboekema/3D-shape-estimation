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

