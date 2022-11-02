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
from parts.data_split import train_val_test_dataloader
from torchvision.transforms import ToTensor


def create_data(gps_error, trees):
    """
    Combining RGB image data and carbon distribution into
    4-channel image for each site.
    """
    
    for site in trees.site.unique():
        print(f"Creating data for site {site}")

        boundary = create_boundary(site, paths["reforestree"])
        img_path = paths["reforestree"] + f"wwf_ecuador/RGB Orthomosaics/{site}.tif"

        #masking the drone image using the boundary
        with rasterio.open(img_path) as raster:
            img, _ = rasterio.mask.mask(raster, boundary, crop=False)

        img = pad(img, patch_size) #padding image to make patches even
        carbon_distribution = compute_carbon_distribution(site, img.shape, trees, gps_error)
        assert img.shape[1:] == carbon_distribution.shape

        img = np.concatenate((img, carbon_distribution))

        np.save(paths["dataset"] + f"{site}", img)




def main():
    # hyperparameters
    patch_size = 400
    angle = 30
    n_rotations = 360 // angle

    #import gps error
    gps_error = get_gps_error()


    paths = {
        "reforestree" : "/home/jan/sem1/ai4good/data/reforestree/",
        "dataset" : "/home/jan/sem1/ai4good/dataset/"
    }

    trees = pd.read_csv(paths["reforestree"] + "field_data.csv")
    trees = trees[["site", "X", "Y", "lat", "lon", "carbon"]]

    create_data(gps_error, trees)

    # train_loader, val_loader, test_loader= train_val_test_dataloader(path_to_dataset, method="across_sites",
    #                                                                  splits=[0.6, 0.2, 0.2], batch_size=16, transform=ToTensor())

    train_loader, val_loader, test_loader= train_val_test_dataloader(paths["dataset"], method="by_site",
                                                                    splits=[4, 1, 1], batch_size=16, transform=ToTensor())