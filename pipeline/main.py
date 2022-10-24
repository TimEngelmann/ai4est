import cv2
import seaborn as sns
import pandas as pd
import numpy as np
import rasterio
from parts.patches import create_upper_left, pad
from parts.boundary import create_boundary
from parts.estimate_carbon import compute_carbon_distribution
from parts.rotate import rotate_distribution, rotate_img
from parts.helper.constants import get_gps_error

# hyperparameters
patch_size = 400
angle = 30
n_rotations = 360 // angle

#import gps error
gps_error = get_gps_error()

# TODO : set path to reforestree folder
path_to_reforestree = "/home/jan/sem1/ai4good/data/reforestree/"
# TODO :  set path where patches will be saved
path_to_dataset = "/home/jan/sem1/ai4good/dataset/"

trees = pd.read_csv(path_to_reforestree + "field_data.csv")
trees = trees[["site", "X", "Y", "lat", "lon", "carbon"]]

def make_imgs_site(img, upper_left):
    """
    For a given site we load the image,
    restrict it with the boundary created from the field data.
    Then the image is padded and split into patches, for
    which we then estimate the carbon stock.
    Finally the image and carbon stock are saved
    together in a compressed numpy file (.npz).
    """

    for j in range(upper_left.shape[0]):
        x_min, y_min = upper_left[j, :]
        patch_img = img[:, x_min:(x_min+patch_size), y_min:(y_min+patch_size)] #restricting to patch
        carbon = carbon_distribution[x_min:(x_min+patch_size),y_min:(y_min+patch_size)].sum()
        assert patch_img.shape == (4, patch_size, patch_size) #sanity check

        yield patch_img

def compute_carbon_site(carbon_distribution, upper_left):
    """
    For a given site we load the image,
    restrict it with the boundary created from the field data.
    Then the image is padded and split into patches, for
    which we then estimate the carbon stock.
    Finally the image and carbon stock are saved
    together in a compressed numpy file (.npz).
    """
    carbon = np.empty(upper_left.shape[0])
    for j in range(upper_left.shape[0]):
        x_min, y_min = upper_left[j, :]
        carbon[j] = carbon_distribution[x_min:(x_min+patch_size),y_min:(y_min+patch_size)].sum()

    return carbon

for site in trees.site.unique():
    print(f"Creating data for site {site}")

    boundary = create_boundary(site, path_to_reforestree)
    img_path = path_to_reforestree + f"wwf_ecuador/RGB Orthomosaics/{site}.tif"

    #masking the drone image using the boundary
    with rasterio.open(img_path) as raster:
        img, _ = rasterio.mask.mask(raster, boundary, crop=False)

    img = pad(img, patch_size) #padding image to make patches even
    carbon_distribution = compute_carbon_distribution(site, img.shape, trees, gps_error)
    assert img.shape[1:] == carbon_distribution.shape

    upper_left = create_upper_left(img.shape, patch_size) #get corners of the patches

    for i in range(n_rotations):
        print(f"Creating patches with rotation angle {i*angle}")
        imgs = make_imgs_site(img, upper_left)
        carbon = compute_carbon_site(carbon_distribution, upper_left)

        sitename_nospaces = site.replace(" ", "")
        path = path_to_dataset + f"{sitename_nospaces}_{i * angle}"
        np.savez_compressed(path, site=site, rotation=i*angle, carbon=carbon, *imgs)

        print("Rotating image")
        img = rotate_img(img, angle)
        carbon_distribution = rotate_distribution(carbon_distribution, angle)
