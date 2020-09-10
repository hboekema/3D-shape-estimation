
import os
import sys
import numpy as np
import cv2
from render_mesh import Mesh
from matplotlib import pyplot as plt
from tqdm import tqdm



def difference_silhouette(silh1, silh2):
    assert silh1.shape == silh2.shape, "shapes must match"
    colours = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 0)
            ]

    silh1_binary = np.invert(silh1 > 0)
    silh2_binary = np.invert(silh2 > 0)
    silh_overlap = silh1_binary.astype(int) + silh2_binary.astype(int) == 2

    silh_comp = np.ones((silh1.shape[0], silh2.shape[1], 3))
    silh_comp[silh1_binary] = colours[0]
    silh_comp[silh2_binary] = colours[1]
    silh_comp[silh_overlap] = colours[2]

    silh_comp *= 255
    return silh_comp.astype(np.uint8)


def grayscale_to_rgb(image, colour="red"):
    rgb_image = np.ones((image.shape[0], image.shape[1], 3))

    if colour == "red":
        colour = (1, 0, 0)
    elif colour == "green":
        colour = (0, 1, 0)

    mask = np.invert(image > 0)
    rgb_image[mask] = colour
    rgb_image *= 255

    return rgb_image.astype(np.uint8)


def get_pc_and_silh(path):
    mesh = Mesh(filepath=path)
    pc = mesh.render_silhouette_with_closing(show=False, closing=False)
    silh = mesh.render_silhouette_with_closing(show=False, closing=True)

    return pc, silh

def show_image(image):
    plt.imshow(image)
    plt.show()


def project_pc(directory_path):
    # Collect .obj files
    directory_files = os.listdir(directory_path)
    sample_filepaths = [f for f in directory_files if ".obj" in f]

    gt_filepaths = sorted([f for f in sample_filepaths if "gt" in f])
    pred_filepaths = sorted([f for f in sample_filepaths if "pred" in f])

    sample_filepairs = zip(gt_filepaths, pred_filepaths)
    #print(sample_filepairs)
    #exit(1)


    # Define colours for GT and predicted point clouds
    gt_colour = (0,255,0)
    pred_colour = (255,0,0)

    print("Rendering and saving visualisations...")
    for filepair in tqdm(sample_filepairs):
        gt_name = filepair[0]
        pred_name = filepair[1]

        gt_path = directory_path + gt_name
        pred_path = directory_path + pred_name

        # Get projected point clouds and silhouettes
        gt_pc, gt_silh = get_pc_and_silh(gt_path)
        pred_pc, pred_silh = get_pc_and_silh(pred_path)

        # Colour the point clouds
        gt_pc_rgb = grayscale_to_rgb(gt_pc, "red")
        pred_pc_rgb = grayscale_to_rgb(pred_pc, "green")

        # Create difference silhouette
        diff_silh = difference_silhouette(gt_silh, pred_silh)

        # Save PC and silhouettes
        gt_save_path = directory_path + "proj_" + gt_name.replace(".obj", ".png")
        pred_save_path = directory_path + "proj_" + pred_name.replace(".obj", ".png")

        cv2.imwrite(gt_save_path, cv2.cvtColor(gt_pc_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(gt_save_path.replace("gt_pc", "gt_silh"), gt_silh)
        cv2.imwrite(pred_save_path, cv2.cvtColor(pred_pc_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(pred_save_path.replace("pred_pc", "pred_silh"), pred_silh)
        cv2.imwrite(pred_save_path.replace("pred_pc", "diff_silh"), cv2.cvtColor(diff_silh, cv2.COLOR_RGB2BGR))

    print("Finished.")


if __name__ == "__main__":
    report_vis_dir = "/data/cvfs/hjhb2/projects/deep_optimiser/report_vis/"

    #subdir = "2d_qualitative/"

    #subdir = "12_joints_ablation/Adam/"
    #subdir = "12_joints_ablation/SGD/"
    #subdir = "12_joints_ablation/deep_opt/"

    #subdir = "losses_ablation_train/regular/"
    #subdir = "losses_ablation_train/update_loss_D20/"
    #subdir = "losses_ablation_train/update_loss_D50/"
    #subdir = "losses_ablation_train/all_losses_D20/"
    #subdir = "losses_ablation_train/all_losses_D50/"

    #subdir = "losses_ablation_test/regular/"
    #subdir = "losses_ablation_test/update_loss/"
    #subdir = "losses_ablation_test/all_losses/"

    #subdir = "Tpose_failures/"

    #subdir = "3d_random_init/deep_opt/"
    #subdir = "3d_random_init/SGD/"
    subdir = "3d_random_init/Adam/"

    example_dir = report_vis_dir + subdir
    project_pc(example_dir)

