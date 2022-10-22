import cv2
import seaborn as sns
import pandas as pd
import numpy as np
import rasterio
from parts.patches import create_upper_left, pad
from parts.boundary import create_boundary
from parts.estimate_carbon import compute_carbon_distribution
from parts.rotate import rotate_distribution, rotate_img

# hyperparameters
patch_size = 400
angle = 30
n_rotations = 360 // angle

#import gps error
gps_error = {"Flora Pluas RGB": [0.25, 0.66],
            "Nestor Macias RGB": [0.6, 0.53],
            "Manuel Macias RGB": [0.69, 0.30],
            "Leonor Aspiazu RGB": [0.47, 0.45],
            "Carlos Vera Arteaga RGB": [0.26, 0.59],
            "Carlos Vera Guevara RGB": [0.27, 0.65]}

# TODO : set path to reforestree folder
path_to_reforestree = "/home/jan/sem1/ai4good/data/reforestree/"
# TODO :  set path where patches will be saved
path_to_dataset = "/home/jan/sem1/ai4good/dataset/"

trees = pd.read_csv(path_to_reforestree + "field_data.csv")
trees = trees[["site", "X", "Y", "lat", "lon", "carbon"]]

for site in trees.site.unique():
    """
    We loop over the sites and for each site load the image,
    restrict it with the boundary created from the field data.
    Then the image is padded and split into patches, for
    which we then estimate the carbon stock.
    Finally the image and carbon stock are saved
    together in a compressed numpy file (.npz).
    """
    print(f"Creating data for site {site}")

    boundary = create_boundary(site, path_to_reforestree)
    img_path = path_to_reforestree + f"wwf_ecuador/RGB Orthomosaics/{site}.tif"

    #masking the drone image using the boundary
    with rasterio.open(img_path) as raster:
        img, _ = rasterio.mask.mask(raster, boundary, crop=False)

    img = pad(img, patch_size) #padding image to make patches even
    carbon_distribution = compute_carbon_distribution(site, img.shape, trees, gps_error)
    upper_left = create_upper_left(img.shape, patch_size) #get corners of the patches

    for i in range(n_rotations):
        print(f"Creating patches with rotation angle {i*angle}")

        for j in range(upper_left.shape[0]):
            x_min, y_min = upper_left[j, :]
            patch_img = img[:, x_min:(x_min+patch_size), y_min:(y_min+patch_size)] #restricting to patch
            carbon = carbon_distribution[x_min:(x_min+patch_size),y_min:(y_min+patch_size)].sum()
            assert patch_img.shape == (4, patch_size, patch_size) #sanity check

            path = path_to_dataset + f"{site} rotation {i * angle}_{j}"
            np.savez(path, img=patch_img, label=carbon, upper_left=upper_left[j,:])

        print("Rotating image")
        img = rotate_img(img, angle)
        carbon_distribution = rotate_distribution(carbon_distribution, angle)
