import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_closing

from smpl_np import SMPLModel
from render_mesh import Mesh

# Generate the silhouettes from the SMPL parameters
smpl = SMPLModel('./keras_rotationnet_v2_demo_for_hidde/./basicModel_f_lbs_10_207_0_v1.0.0.pkl')

# Artificial sample data
sample_pose = 0.0 * (np.random.rand(smpl.pose_shape[0], smpl.pose_shape[1]) - 0.5)
sample_beta = 0.0 * (np.random.rand(smpl.beta_shape[0]) - 0.5)
sample_trans = np.zeros(smpl.trans_shape[0])
#sample_trans = 0.1 * (np.random.rand(smpl.trans_shape[0]) - 0.5)

sample_y = np.array([sample_pose.ravel(), sample_beta, sample_trans])
sample_pc = smpl.set_params(sample_pose, sample_beta, sample_trans)

#pc_subset = sample_pc[5020:5150]
#pc_subset = sample_pc[5400:5600]
#pc_subset = sample_pc[5020:5600]
#pc_subset = sample_pc[1850:2100]
pc_subset = sample_pc[[1850, 1600, 2050, 5350, 5050, 5500]]

def render_silhouette(pc, dim=(256, 256), morph_mask=None, show=True, title="silhouette"):
    """ Create a(n orthographic) silhouette out of a 2D slice of the pointcloud """
    x_sf = dim[0] - 1
    y_sf = dim[1] - 1

    # Collapse the points onto the x-y plane by dropping the z-coordinate
    verts = pc
    mesh_slice = verts[:, :2]
    mesh_slice[:, 0] += 1
    mesh_slice[:, 1] += 1.2
    mesh_slice *= np.mean(dim)/2
    coords = np.round(mesh_slice).astype("uint8")

    # Create background to project silhouette on
    image = np.ones(shape=dim, dtype="uint8")
    for coord in coords:
        # Fill in values with a silhouette coordinate on them
        image[y_sf-coord[1], coord[0]] = 0

    # Finally, perform a morphological closing operation to fill in the silhouette
    if morph_mask is None:
        # Use a circular mask as the default operator
        morph_mask = np.array([[0.34, 0.34, 0.34],
                                   [0.34, 1.00, 0.34],
                                   [0.34, 0.34, 0.34]
                                   ])
    #image = np.invert(image.astype(bool))
    #image = np.invert(binary_closing(image, structure=morph_mask, iterations=1)).astype(np.uint8)
    image *= 255

    if show:
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.show()

    return image

render_silhouette(pc_subset)


