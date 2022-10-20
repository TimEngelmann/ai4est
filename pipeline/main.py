import cv2
import seaborn as sns
import pandas as pd
import numpy as np
import rasterio
from parts.patches import make_grid, pad, convert_coordinates_to_df, remove_out_of_bounds
from parts.boundary import create_boundary
from parts.estimate_agb import estimate_agb
from parts.rotate import rotate_grid, rotate_img

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
    grid_coords = make_grid(img.shape, patch_size) #get corners of the patches

    for i in range(n_rotations):
        print(f"Creating patches with rotation angle {i*angle}")
        #rotate the grid
        rotated_coords = rotate_grid(grid_coords, angle, img.shape)
        filtered_coords = remove_out_of_bounds(rotated_coords, img.shape)
        patches = convert_coordinates_to_df(filtered_coords, site) #convert grid array to patches df
        patches = estimate_agb(patches, trees, gps_error)

        #rotate the image

        for j, patch in patches.iterrows():
            x_min, y_min = patch["vertices"].astype(int)[0,:]
            patch_img = img[:, x_min:(x_min+patch_size), y_min:(y_min+patch_size)] #restricting to patch
            assert patch_img.shape == (4, patch_size, patch_size) #sanity check

            path = path_to_dataset + f"{site} rotation {i * angle}_{j}"
            np.savez(path, img=patch_img, label=patch["carbon"], vertices=patch["vertices"]) #saving data

        print("Rotating image")
        img = rotate_img(img, angle)
