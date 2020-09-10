
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_closing

sys.path.append("/data/cvfs/hjhb2/projects/deep_optimiser/code/")
from render_mesh import Mesh


def tight_crop(img, invert=False):
    """ Crop image of silhouette tightly (i.e. boundaries of image are at outermost edge of silhouette """
    if invert:
        img = (img == 0)
    x_nonzero_indices = np.nonzero(img)[0]
    x_min = np.min(x_nonzero_indices)
    x_max = np.max(x_nonzero_indices)
    y_nonzero_indices = np.nonzero(img)[1]
    y_min = np.min(y_nonzero_indices)
    y_max = np.max(y_nonzero_indices)

    cropped_img = img[x_min:x_max+1, y_min:y_max+1]

    return cropped_img


def pad_minor_axis(img):
    """ Pad minor axis of image to match major axis """
    img_np = np.array(img)
    wd = img_np.shape[1]
    ht = img_np.shape[0]

    ww = np.max([wd, ht])
    hh = np.max([wd, ht])
    padded_img = np.zeros((ww, hh))

    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    padded_img[yy:yy+ht, xx:xx+wd] = img

    return padded_img


def normalise_img(img, dim=(128, 128)):
    """ Normalise input image and output with given dimensions """
    wd = int(np.floor(0.90 * dim[0]))
    ht = int(np.floor(0.90 * dim[1]))

    img_cropped = tight_crop(img)
    img_padded = pad_minor_axis(img_cropped)
    img_resized = cv2.resize(1.0*img_padded, (wd, ht), interpolation=cv2.INTER_NEAREST)

    xx = (dim[0] - wd) // 2
    yy = (dim[1] - ht) // 2
    normalised_img = np.zeros(dim)

    normalised_img[yy:yy+ht, xx:xx+wd] = img_resized
    normalised_img = 255 * normalised_img.astype(np.uint8)

    return normalised_img

def augment_image(img):
    xx = np.random.randint(-1, 2)
    yy = np.random.randint(-1, 2)

    if xx == 1:
        shifted_img = np.insert(img, -1, np.zeros_like(img[-1]), axis=0)
        shifted_img = shifted_img[1:]
    elif xx == -1:
        shifted_img = np.insert(img, 0, np.zeros_like(img[-1]), axis=0)
        shifted_img = shifted_img[:-1]
    else:
        shifted_img = img


    if yy == 1:
        shifted_img = np.insert(shifted_img, -1, np.zeros_like(shifted_img[:, -1]), axis=1)
        shifted_img = shifted_img[:, 1:]
    elif yy == -1:
        shifted_img = np.insert(shifted_img, 0, np.zeros_like(shifted_img[-1]), axis=1)
        shifted_img = shifted_img[:, :-1]
    else:
        shifted_img = img

    dilate = np.random.randint(0, 2)
    dilate = 1
    if dilate == 1:
        morph_mask = np.array([[0.34, 0.34, 0.34],
                                   [0.34, 1.00, 0.34],
                                   [0.34, 0.34, 0.34]
                                   ])
        new_img = binary_closing(shifted_img != 0, structure=morph_mask, iterations=1).astype(np.uint8)
        new_img *= 255
    else:
        new_img = shifted_img

    return new_img



if __name__ == "__main__":
    mesh_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/example_meshes/"
    obj_paths = os.listdir(mesh_dir)
    for obj_path in obj_paths:
        mesh = Mesh(os.path.join(mesh_dir, obj_path))
        silh = mesh.render_silhouette(dim=[256, 256], show=True)
        normalised_silh = normalise_img(silh, dim=(128, 128))

        #plt.imshow(silh_cropped, cmap="gray")
        plt.imshow(normalised_silh, cmap="gray")
        plt.show()

        augmented_silh = augment_image(normalised_silh)

        plt.imshow(augmented_silh, cmap="gray")
        plt.show()

