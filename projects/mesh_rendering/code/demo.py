from argparse import ArgumentParser
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
from keras.models import Model
from encoder import Encoder
from encoder import SimpleEncoderArchitecture

from smpl_tf import smpl_model
from smpl_np import SMPLModel
from render_mesh import Mesh

parser = ArgumentParser()
parser.add_argument("--model", help="Path to the model weights")
parser.add_argument("--data", help="Path to the test image")
args = parser.parse_args()

einfo={'model_weights_path':'../models/model.01-0.07 2019-11-01_13:30:30.hdf5'}
# # Ensure that TF2.0 is not used
tf.disable_v2_behavior()
tf.enable_eager_execution()

smpl = SMPLModel('../SMPL/model.pkl')
def get_random_sample(smpl):
    pose = 0.65 * (np.random.rand(*smpl.pose_shape) - 0.5)
    beta = 0.06 * (np.random.rand(*smpl.beta_shape) - 0.5)
    trans = np.zeros(smpl.trans_shape)

    parameters = np.concatenate([pose.ravel(), beta, trans])

    # Create the body mesh
    pointcloud = smpl.set_params(beta=beta, pose=pose, trans=trans)

    # Render the silhouette
    mesh = Mesh(pointcloud=pointcloud)
    mesh.faces = smpl.faces
    silhouette = mesh.render_silhouette(dim=(256, 256), show=False)

    return silhouette, mesh, parameters

if args.data is not None:
    # Load data
    mesh = None
    silhouette = cv2.imread(args.data, cv2.IMREAD_GRAYSCALE)
else:
    # Generate a mesh and its silhouette
    silhouette, mesh, _ = get_random_sample(smpl)

random_sample, _, parameters = get_random_sample(smpl)

# Load model
# encoder = Encoder()
#encoder.train_step(random_sample.reshape((1, *random_sample.shape, 1)), parameters.reshape(1, *parameters.shape))
# encoder.load_weights(args.model)
encoder_inputs,encoder_outputs = SimpleEncoderArchitecture((256,256,1))
encoder = Model(inputs=encoder_inputs,outputs=encoder_outputs)
encoder.summary()

encoder.load_weights(einfo['model_weights_path'])
# Predict the parameters from the silhouette and generate a mesh
prediction = encoder.predict(silhouette.reshape(1, 256, 256, 1))
prediction = tf.cast(prediction, tf.float64)
print("Shape of predictions:'"+str(prediction.shape))
print(prediction[0,82:85])

pred_pc, faces = smpl_model("../model.pkl", prediction[0,72:82]*0+1, prediction[0,:72], prediction[0,82:85]*0)
# Render the mesh
pred_mesh = Mesh(pointcloud=pred_pc.numpy())
pred_mesh.faces = faces

if mesh is not None:
    mesh.render3D()

print("Rendering prediction")
pred_mesh.render3D()

# Now render their silhouettes
# cv2.imshow("True silhouette", silhouette)
# pred_mesh.render_silhouette(title="Predicted silhouette")
