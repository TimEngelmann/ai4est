import cv2
import seaborn as sns
import pandas as pd
import numpy as np
import rasterio
from parts.patches import make_grid, pad
from parts.boundary import create_boundary
from parts.estimate_agb import estimate_agb

# hyperparameters
patch_size = 400

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
    boundary = create_boundary(site, path_to_reforestree)
    img_path = path_to_reforestree + f"wwf_ecuador/RGB Orthomosaics/{site}.tif"

    #masking the drone image using the boundary
    with rasterio.open(img_path) as raster:
        img, _ = rasterio.mask.mask(raster, boundary, crop=True)

    padded_img = pad(img, patch_size) #padding image to make patches even
    patches = make_grid(site, padded_img.shape, patch_size) #get corners of the patches
    patches = estimate_agb(patches, trees, gps_error) #compute carbon for the patches

    for i,patch in patches.iterrows():
        x_min, y_min = patch["vertices"][0,:]
        patch_img = padded_img[:, x_min:(x_min+patch_size), y_min:(y_min+patch_size)] #restricting to patch
        assert patch_img.shape == (4, patch_size, patch_size) #sanity check

        path = path_to_dataset + f"{site} {i}"
        np.savez(path, img=patch_img, label=patch["carbon"]) #saving data
