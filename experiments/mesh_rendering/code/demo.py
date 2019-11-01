from argparse import ArgumentParser
import numpy as np
import tensorflow.compat.v1 as tf
import cv2

from encoder import Encoder
from smpl_np import SMPLModel
from render_mesh import Mesh

parser = ArgumentParser()
parser.add_argument("model", help="Path to the model weights")
parser.add_argument("--data", help="Path to the test image")
args = parser.parse_args()

# Ensure that TF2.0 is not used
tf.disable_v2_behavior()
tf.enable_eager_execution()

# Load model
encoder = Encoder()
encoder.load_weights(args.model)

smpl = SMPLModel('../SMPL/model.pkl')
if args.data is not None:
    # Load data
    mesh = None
    silhouette = cv2.imread(args.data, cv2.IMREAD_GRAYSCALE)
else:
    # Generate a mesh and its silhouette
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

# Predict the parameters from the silhouette and generate a mesh
prediction = encoder.predict_(silhouette)
pred_pc = smpl.set_params(beta=prediction[:72], pose=prediction[72:82], trans=prediction[82:])

# Render the mesh
pred_mesh = Mesh(pointcloud=pred_pc)
pred_mesh.faces = smpl.faces

if mesh is not None:
    mesh.render3D()

pred_mesh.render3D()

# Now render their silhouettes
cv2.imshow("True silhouette", silhouette)
pred_mesh.render_silhouette(title="Predicted silhouette")
