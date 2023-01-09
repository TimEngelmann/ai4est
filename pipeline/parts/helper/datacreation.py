import numpy as np
import logging
import rasterio
from parts.patches import pad
from parts.boundary import create_boundary
from parts.estimate_carbon import compute_carbon_distribution
from PIL import Image
import os


def create_data(paths, hyperparameters, trees):
    """
    Combining RGB image data and carbon distribution into
    4-channel image for each site.
    """

    patch_size = hyperparameters["patch_size"]

    carbon_threshold = 50
    if hyperparameters["carbon_threshold"]:
        carbon_threshold = hyperparameters["carbon_threshold"]

    tree_density = False
    if hyperparameters["tree_density"]:
        tree_density = hyperparameters["tree_density"]

    boundary_shape = "convex_hull"
    if hyperparameters["boundary_shape"]:
        boundary_shape = hyperparameters["boundary_shape"]


    # creating folder "paths["dataset"]/processed" if it doesn't exist
    if not os.path.exists(paths["dataset"] + "sites"):
        logging.info("Creating directory %s", paths["dataset"] + "sites")
        os.mkdir(paths["dataset"] + "sites")

    for site in trees.site.unique():
        logging.info("Creating data for site %s", site)

        #get covariance for normal distribution
        if isinstance(hyperparameters["covariance"], dict):
            covariance = np.array(hyperparameters["covariance"][site])
        else:
            covariance = np.array(hyperparameters["covariance"])
        mean = np.array(hyperparameters["mean"])

        boundary = create_boundary(site, paths["reforestree"], shape=boundary_shape)
        img_path = paths["reforestree"] + f"wwf_ecuador/RGB Orthomosaics/{site}.tif"

        #masking the drone image using the boundary
        with rasterio.open(img_path) as raster:
            img, _ = rasterio.mask.mask(raster, boundary, crop=False)

        img = pad(img, patch_size) #padding image to make patches even
        carbon_distribution = compute_carbon_distribution(site, img.shape, trees, mean, covariance, carbon_threshold, tree_density)
        assert img.shape[1:] == carbon_distribution.shape


        np.save(paths["dataset"] + "sites/" + f"{site}_carbon", carbon_distribution)
        im = Image.fromarray(np.moveaxis(img, 0, -1)[:,:,:3])
        im.save(paths["dataset"] + "sites/" + f"{site}_image.png")
