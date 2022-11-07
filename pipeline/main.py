import cv2
import seaborn as sns
import pandas as pd
import numpy as np
import rasterio
from parts.patches import create_upper_left, pad
from parts.boundary import create_boundary
from parts.estimate_carbon import compute_carbon_distribution
from parts.helper.constants import get_gps_error
from parts.processing import process
from parts.data_split import train_val_test_dataloader
from torchvision.transforms import ToTensor
import argparse


#argument parser
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#TODO: comment this section out to run the code without an argparser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--createpatches", required=True, type=str2bool,
	help="boolean for whether to create the patches images dataset")
ap.add_argument("-b", "--batchsize", required=False, type=int,
                default=64, help="batch size for dataloader")
ap.add_argument("-s", "--splitting", nargs='+', required=False, type=float,
                default=[4,1,1], help="list of length 3 [size_train, size_val, size_test]. "
                                      "If summing up to 1, the data will be split randomly across each site,"
                                      "if summing up to 6, the data data will be split by site")
args = ap.parse_args()

create_dataset= args.createpatches
splits=args.splitting
batch_size= args.batchsize


def create_data(paths, gps_error, trees, hyperparameters):
    """
    Combining RGB image data and carbon distribution into
    4-channel image for each site.
    """

    patch_size = hyperparameters["patch_size"]

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

        img = np.concatenate((img, carbon_distribution.reshape(1, img.shape[1], img.shape[2])))

        np.save(paths["dataset"] + f"{site}", img)




def main():

    # hyperparameters
    hyperparameters = {
        "patch_size" : 400,
        "angle" : 30,
        "rotations" : [0, 30, 60]  # 360 // 30
    }

    #import gps error
    gps_error = get_gps_error()


    paths = {
        "reforestree" : "../data/reforesTree/",
# path_to_reforestree = "~/ai4est/data/reforestree/"
        "dataset" : ".././data/dataset/"
# path_to_dataset = "~/ai4est/data/dataset/"
    }

    trees = pd.read_csv(paths["reforestree"] + "field_data.csv")
    trees = trees[["site", "X", "Y", "lat", "lon", "carbon"]]
    if create_dataset:
        create_data(paths, gps_error, trees, hyperparameters)


    data = process(trees.site.unique(), hyperparameters, paths)
    train_loader, val_loader, test_loader= train_val_test_dataloader(paths["dataset"],
                                                                 splits=splits, batch_size=batch_size, transform=ToTensor())


main()
