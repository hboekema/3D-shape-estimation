
import sys
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import keras.backend as K
#import segmentation_models as sm
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
#from render_mesh import Mesh
from keras.applications.resnet50 import ResNet50


# Define custom loss functions
def false_loss(y_true, y_pred):
    """ Loss function that simply return the predicted value (i.e. target is 0) """
    return y_pred

def no_loss(y_true, y_pred):
    return tf.reduce_sum(tf.multiply(y_true, 0))

# Helper functions
def render_orth(verts, dim=(256, 256, 1), morph_mask=None):
    """ Orthographic rendering of points """
    x_sf = dim[0] - 1
    y_sf = dim[1] - 1

    # Collapse the points onto the x-y plane by dropping the z-coordinate
    mesh_slice = Lambda(lambda x: x[:, :2])(verts)
    mesh_shape = mesh_slice.shape
    print("mesh_slice shape: " + str(mesh_slice.shape))
    displacement = np.ones(shape=(mesh_shape[1], mesh_shape[2]))
    displacement[:, 1] = 1.2
    displacement = K.constant(displacement)
    print("displacement (constant) shape: " + str(displacement.shape))

    mesh_slice = Lambda(lambda x: x[0] + x[1])([mesh_slice, displacement])
    #mesh_slice = Lambda(lambda x: x[:, 0] + 1)(mesh_slice)
    #mesh_slice = Lambda(lambda x: x[:, 1] + 1.2)(mesh_slice)
    zoom = np.mean(dim)/2.2     # zoom into the projected pc
    zoom = K.constant(zoom)
    print("zoom (constant) shape: " + str(zoom.shape))
    mesh_slice = Lambda(lambda x: x[0] * x[1])([mesh_slice, zoom])

    # Flip the y-axis
    flip_disp = np.zeros(shape=(mesh_shape[1], mesh_shape[2]), dtype="uint8")
    flip_disp[:, 1] = -y_sf
    flip_disp = K.constant(flip_disp)
    print("flip_disp shape: " + str(flip_disp.shape))
    flip = np.ones(shape=(mesh_shape[1], mesh_shape[2]))
    flip[:, 1] = -1
    flip = K.constant(flip)
    print("flip shape: " + str(flip.shape))
    coords = Lambda(lambda x: K.tf.cast(K.tf.round(x), K.tf.uint8))(mesh_slice)
    coords = Lambda(lambda x: x[0] + x[1])([coords, flip_disp])
    coords = Lambda(lambda x: x[0] * x[1])([coords, flip])
    coords = Lambda(lambda x: K.tf.reverse(x, axis=-1))(coords)

    # Create background to project silhouette on
    image = K.ones(shape=dim, dtype="uint8")
    image[coords] = 0
    #Lambda(lambda x: x[0][x[1]] = 0)([image, coords])

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


        image = Reshape(target_shape=(image_shape[0]. image_shape[1]))(image)
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


# Custom architectures
def SimpleArchitecture(input_shape):
    """ Basic model for predicting 3D human pose and shape from an input silh """
    # The segmented silhouette is the model input
    input_silh = Input(shape=input_shape, name="input_silh")

    resnet_model = ResNet50(include_top=False, input_tensor=input_silh, input_shape=input_shape, weights=None)
    encoder_architecture = Flatten()(resnet_model.outputs[0])
    encoder_params = Dense(85, activation="tanh")(encoder_architecture)

    # Load SMPL model and get necessary parameters
    smpl_params = load_params('./keras_rotationnet_v2_demo_for_hidde//basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    _, _, input_info = get_parameters()
    input_betas = Lambda(lambda x: x[:, 72:82])(encoder_params)
    input_pose_rodrigues = Lambda(lambda x: x[:, 0:72])(encoder_params)
    input_trans = Lambda(lambda x: x[:, 82:85])(encoder_params)

    encoder_mesh = Points3DFromSMPLParams(input_betas, input_pose_rodrigues, input_trans, smpl_params, input_info)

    # Render the silhouette orthographically
    decoder_silh = render_orth(encoder_mesh)

    return [input_silh], [encoder_params, encoder_mesh, decoder_silh]





