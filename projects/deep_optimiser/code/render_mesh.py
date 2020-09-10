import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from copy import copy
from PIL import Image, ImageOps
from scipy.ndimage.morphology import binary_closing
import trimesh
import cv2


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

        with open(filepath, 'r') as file_:
            # Read the file a line at a time, identifying vertices and faces
            for line in file_:
                if line[0] == 'v':
                    self.verts.append([float(vi) for vi in line[2:].split(' ')])
                elif line[0] == 'f':
                    self.faces.append([int(fi) - 1 for fi in line[2:].split(' ')])
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

    def render_silhouette_pers(self, dim=(256, 256), cam_zdist=1, morph_mask="default", show=True, title="silhouette"):
        """ Render silhouette by projecting (taking account of perspective) the point cloud """
        # Define the scale factors
        x_sf = dim[0] - 1
        y_sf = dim[1] - 1

        # Collapse the points onto the x-y plane by dropping the z-coordinate, adjusting for the camera position (in distance in z-coordinate from projective plane)
        verts = copy(self.verts)
        verts_z = verts[:, 2]
        mesh_slice = verts[:, :2]

        mesh_slice[:, 0] *= x_sf/verts_z * cam_zdist
        mesh_slice[:, 1] *= y_sf/verts_z * cam_zdist

        coords = np.round(mesh_slice).astype("uint8")

        # Create background to project silhouette on
        image = np.ones(shape=dim, dtype="uint8")
        for coord in coords:
            # Fill in values with a silhouette coordinate on them
            x = coord[1]
            y = coord[0]

            if x <= x_sf and y <= y_sf:
                # Only show the silhouette if it is visible
                image[x, y] = 0

        if morph_mask is not None:
            # Optionally apply a morphological closing operation to fill in the silhouette
            if morph_mask == "default":
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

    def render_silhouette(self, dim=(256, 256), morph_mask=None, show=False, title="silhouette"):
        """ Create a(n orthographic) silhouette out of a 2D slice of the pointcloud """
        x_sf = dim[0] - 1
        y_sf = dim[1] - 1

        # Collapse the points onto the x-y plane by dropping the z-coordinate
        verts = copy(self.verts)
        mesh_slice = verts[:, :2]
        mesh_slice[:, 0] += 1
        mesh_slice[:, 1] += 1.2
        mesh_slice *= np.mean(dim)/2.2
        coords = np.round(mesh_slice)
        x_mask = np.logical_and(coords[:, 0] <= x_sf, coords[:, 0] >= 0)
        y_mask = np.logical_and(coords[:, 1] <= y_sf, coords[:, 1] >= 0)
        mask = x_mask * y_mask
        coords = coords[mask]
        coords =  coords.astype("uint8")

        # Create background to project silhouette on
        image = np.ones(shape=dim, dtype="uint8")
        #for coord in coords:
        #    # Fill in values with a silhouette coordinate on them
        #    image[y_sf-coord[1], coord[0]] = 0
        image[y_sf-coords[:,1], coords[:,0]] = 0
        #exit(1)

        # Finally, perform a morphological closing operation to fill in the silhouette
        if morph_mask is None:
            # Use a circular mask as the default operator
            morph_mask = np.array([[0.34, 0.34, 0.34],
                                   [0.34, 1.00, 0.34],
                                   [0.34, 0.34, 0.34]
                                   ])

        iterations = int(np.floor(dim[0]/128))
        image = np.invert(image.astype(bool))

        if False:
            #image_x2 = image[::4, ::4]
            image_x2 = image[::2, ::2]
            #image_x2 = image
            image_x2_morphed = np.invert(binary_closing(image_x2, structure=morph_mask, iterations=iterations)).astype(np.uint8)
            image = cv2.resize(1.0*image_x2_morphed, (dim[0], dim[1]), interpolation=cv2.INTER_NEAREST)
        else:
            image = image.astype(np.uint8)
        image *= 255

        if show:
            plt.imshow(image, cmap="gray")
            plt.title(title)
            plt.show()

        return image

    def render_silhouette_with_closing(self, dim=(256, 256), morph_mask=None, show=False, title="silhouette", closing=True):
        """ Create a(n orthographic) silhouette out of a 2D slice of the pointcloud """
        x_sf = dim[0] - 1
        y_sf = dim[1] - 1

        # Collapse the points onto the x-y plane by dropping the z-coordinate
        verts = copy(self.verts)
        mesh_slice = verts[:, :2]
        mesh_slice[:, 0] += 1
        mesh_slice[:, 1] += 1.2
        mesh_slice *= np.mean(dim)/2.2
        coords = np.round(mesh_slice)
        x_mask = np.logical_and(coords[:, 0] <= x_sf, coords[:, 0] >= 0)
        y_mask = np.logical_and(coords[:, 1] <= y_sf, coords[:, 1] >= 0)
        mask = x_mask * y_mask
        coords = coords[mask]
        coords =  coords.astype("uint8")

        # Create background to project silhouette on
        image = np.ones(shape=dim, dtype="uint8")
        #for coord in coords:
        #    # Fill in values with a silhouette coordinate on them
        #    image[y_sf-coord[1], coord[0]] = 0
        image[y_sf-coords[:,1], coords[:,0]] = 0
        #exit(1)

        # Finally, perform a morphological closing operation to fill in the silhouette
        if morph_mask is None:
            # Use a circular mask as the default operator
            morph_mask = np.array([[0.34, 0.34, 0.34],
                                   [0.34, 1.00, 0.34],
                                   [0.34, 0.34, 0.34]
                                   ])

        iterations = int(np.floor(dim[0]/128))

        if closing:
            image = np.invert(image.astype(bool))
            #image_x2 = image[::4, ::4]
            #image_x2 = image[::2, ::2]
            image_x2 = image
            image_x2_morphed = np.invert(binary_closing(image_x2, structure=morph_mask, iterations=iterations))
            #image = np.invert(image_x2_morphed).astype(np.uint8)
            image = image_x2_morphed.astype(np.uint8)
            image = cv2.resize(1.0*image, (dim[0], dim[1]), interpolation=cv2.INTER_NEAREST)
        else:
            image = image.astype(np.uint8)

        image *= 255

        if show:
            plt.imshow(image, cmap="gray")
            plt.title(title)
            plt.show()

        return image

    def render_silhouette_centre(self, dim=(256, 256), padding=0.00, morph_mask=None, show=True, title="silhouette"):
        """ Create a(n orthographic) silhouette out of a 2D slice of the pointcloud """
        # Define the scale factors and padding
        x_sf = dim[0] - 1
        y_sf = dim[1] - 1

        # Collapse the points onto the x-y plane by dropping the z-coordinate
        verts = copy(self.verts)
        mesh_slice = verts[:, :2]

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
    mesh_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/example_meshes/"
    obj_paths = os.listdir(mesh_dir)
    for obj_path in obj_paths:
        mesh = Mesh(os.path.join(mesh_dir, obj_path))
        mesh.render_silhouette(dim=[256, 256], show=True)
        #mesh.render3D()

