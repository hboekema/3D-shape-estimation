import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.ndimage.morphology import binary_closing
import trimesh

matplotlib.use("Qt5Agg")

class Mesh:
    def __init__(self, filepath=None, pointcloud=None):
        if pointcloud is not None:
            self.verts = pointcloud.astype("float32")
            self.faces = None

        elif filepath is not None:
            self.verts = []
            self.faces = []
            self.load_mesh(filepath)

    def load_mesh(self, filepath):
        """ Store vertices and faces of the mesh """
        self.verts = []
        self.faces = []

        with open(filepath, 'r') as file:
            # Read the file a line at a time, identifying vertices and faces
            for line in file:
                if line[0] == 'v':
                    self.verts.append([float(vi) for vi in line[2:].split(sep=' ')])
                elif line[0] == 'f':
                    self.faces.append([int(fi) - 1 for fi in line[2:].split(sep=' ')])
                else:
                    pass

            self.verts = np.array(self.verts, dtype="float32")
            self.faces = np.array(self.faces, dtype=int)

    def __repr__(self):
        """ Represent object as the arrays of its vertices and faces """
        return str({"Vertices": self.verts, "Faces": self.faces})

    def rotate(self, theta=(0, 0, 0), save=True):
        # Construct the rotation matrices
        theta_x = np.radians(theta[1])
        theta_y = np.radians(theta[2])
        theta_z = np.radians(theta[0])

        R_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        R_y = np.array([[np.cos(theta_y), 0, -np.sin(theta_y)], [0, 1, 0], [np.sin(theta_z), 0, np.cos(theta_z)]])
        R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
        R = np.matmul(R_z, R_y, R_x)

        new_verts = np.array([np.dot(R, vert) for vert in self.verts])
        if save:
            self.verts = new_verts

        return new_verts

    def render3D(self):
        """ Render the mesh in 3D """
        mesh = trimesh.Trimesh(vertices=self.verts, faces=self.faces)
        mesh.show(resolution=(512, 512))

    def render_silhouette(self, dim=(256, 256), padding=0.15, morph_mask=None, show=True, title="silhouette"):
        """ Create a silhouette out of a 2D slice of the pointcloud """
        # Define the scale factors and padding
        x_sf = dim[0] - 1
        y_sf = dim[1] - 1

        # Collapse the points onto the x-y plane by dropping the z-coordinate
        mesh_slice = self.verts[:, :2]

        # Adjust the x-y coordinates by shifting appropriately (considering padding)
        mesh_slice[:, 0] -= np.min(mesh_slice[:, 0]) - padding
        mesh_slice[:, 1] -= np.min(mesh_slice[:, 1]) - padding

        # Scale the array such that the largest value is 1.0 (then pad)
        array_max = np.max([np.max(mesh_slice[:, 0]), np.max(mesh_slice[:, 1])])
        mesh_slice[:, 0] = ((1 + padding/2) + mesh_slice[:, 0] / array_max) * x_sf
        mesh_slice[:, 1] = ((1 + padding/2) - (mesh_slice[:, 1] / array_max)) * y_sf
        mesh_slice[:, 0] += ((1 - padding) - np.max(mesh_slice[:, 0]))/2
        mesh_slice[:, 1] += ((1 - padding) - np.min(mesh_slice[:, 1]))/2

        # Convert to uint8 coordinate values and shift such that the image is roughly central
        coords = np.round(mesh_slice).astype("uint8")

        # Create background to project silhouette on
        image = np.ones(shape=dim, dtype="uint8")
        for coord in coords:
            # Fill in values with a silhouette coordinate on them
            image[coord[1], coord[0]] = 0

        # Finally, perform a morphological closing operation to fill in the silhouette
        if morph_mask is None:
            # Use a circular mask as the default operator
            morph_mask = np.array([[0.34, 0.34, 0.34],
                                   [0.34, 1.00, 0.34],
                                   [0.34, 0.34, 0.34]
                                   ])
        image = np.invert(image.astype(bool))
        image = np.invert(binary_closing(image, structure=morph_mask, iterations=2)).astype(np.uint8)
        image *= 255

        if show:
            plt.imshow(image, cmap="gray")
            plt.title(title)
            plt.show()

        return image

    def render2D(self, theta=None, save_path=None, show=True):
        """ Create a 2D rendering of a slice of the mesh """
        if theta is not None:
            verts = self.rotate(theta, save=False)
        else:
            verts = self.verts

        # Collapse the points onto the x-y plane by dropping the z-coordinate
        mesh_slice = verts[:, :2]

        # Use pyplot to plot the resulting points
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.scatter(mesh_slice[:, 0], mesh_slice[:, 1])
        ax.set_aspect("equal", adjustable="datalim")

        if save_path is not None:
            fig.savefig(save_path, format="png", cmap="gray")
            image = Image.open(save_path).convert("L")
            image = ImageOps.fit(image, (128, 128), Image.ANTIALIAS)
            image.save(save_path, format="png")

            plt.close()

            return image
        elif show:
            plt.show()

        plt.close()

if __name__ == "__main__":
    mesh_dir = "../example_meshes/"
    obj_paths = os.listdir(mesh_dir)
    for obj_path in obj_paths:
        mesh = Mesh(os.path.join(mesh_dir, obj_path))
        mesh.render3D()
        mesh.render_silhouette()
